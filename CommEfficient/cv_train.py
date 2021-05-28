import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
import math
import os
import time
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

import models
from fed_aggregator import FedModel, FedOptimizer
from utils import make_logdir, union, Timer, TableLogger, parse_args
from utils import PiecewiseLinear, Exp, num_classes_of_dataset, steps_per_epoch
from data_utils import FedSampler, FedCIFAR10, FedImageNet, FedCIFAR100, FedFEMNIST, FedFashionMNIST
from data_utils import fmnist_train_transforms, fmnist_test_transforms
from data_utils import cifar10_train_transforms, cifar10_test_transforms
from data_utils import cifar100_train_transforms, cifar100_test_transforms
from data_utils import imagenet_train_transforms, imagenet_val_transforms
from data_utils import femnist_train_transforms, femnist_test_transforms

import torch.multiprocessing as multiprocessing

#from line_profiler import LineProfiler
#import atexit
#profile = LineProfiler()
#atexit.register(profile.print_stats)

# module for computing accuracy
class Correct(torch.nn.Module):
    def forward(self, classifier, target):
        return (classifier.max(dim = 1)[1] == target).float().mean()

def criterion_helper(outputs, target, lam):
    ce = -F.log_softmax(outputs, dim=1)
    mixed = torch.zeros_like(outputs).scatter_(
                1, target.data.view(-1, 1), lam.view(-1, 1)
            )
    return (ce * mixed).sum(dim=1).mean()

def mixup_criterion(outputs, y_a, y_b, lam):
    return (criterion_helper(outputs, y_a, lam)
            + criterion_helper(outputs, y_b, 1 - lam))

# whether args.grad_reduction is median or mean,
# each worker still means gradients locally
ce_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

accuracy_metric = Correct()

def compute_loss_mixup(model, batch, args):
    images, targets = batch
    inputs, targets_a, targets_b, lam = mixup_data(
            images, targets, args.mixup_alpha,
            use_cuda="cuda" in args.device
        )
    outputs = model(inputs)
    loss = mixup_criterion(outputs, targets_a, targets_b, lam)
    pred = torch.max(outputs, 1)[1]
    correct = (lam * pred.eq(targets_a)
               + (1 - lam) * pred.eq(targets_b)).float().sum()
    accuracy = correct / targets.size()[0]
    return loss, accuracy

def compute_loss_ce(model, batch, args):
    images, targets = batch
    pred = model(images)
    loss = ce_criterion(pred, targets)
    accuracy = accuracy_metric(pred, targets)
    return loss, accuracy

def compute_loss_train(model, batch, args):
    """
    if args.do_mixup:
        return compute_loss_mixup(model, batch, args)
    else:
    """
    return compute_loss_ce(model, batch, args)

def compute_loss_mal(model, batch, args):
    loss, accuracy = compute_loss_ce(model, batch, args)
    boosted_loss = torch.tensor(args.mal_boost).to(args.device) * loss
    #print(f"Mal update on batch of size {batch[0].size()} with boosted loss {boosted_loss.mean()} and acc {accuracy}")
    return boosted_loss, accuracy

def compute_loss_val(model, batch, args):
    return compute_loss_ce(model, batch, args)

def train(model, opt, lr_scheduler, train_loader, test_loader,
          args, writer, loggers=(), timer=None, mal_loader=None):
    timer = timer or Timer()

    if args.eval_before_start:
        if args.do_malicious:
            # mal
            mal_loss, mal_acc = run_batches(
                    model, None, None, mal_loader, False, 1, 1, args
                )
            print("Mal acc at epoch 0: {:0.4f}".format(mal_acc))
            # val
            test_loss, test_acc = run_batches(
                    model, None, None, test_loader, False, 1, 1, args
                )
            print("Test acc at epoch 0: {:0.4f}".format(test_acc))

    # ceil in case num_epochs in case we want to do a
    # fractional number of epochs
    for epoch in range(math.ceil(args.num_epochs)):
        epoch_stats = {}
        if epoch == math.ceil(args.num_epochs) - 1:
            epoch_fraction = args.num_epochs - epoch
        else:
            epoch_fraction = 1
        # train
        train_loss, train_acc = run_batches(
                model, opt, lr_scheduler, train_loader,
                True, epoch_fraction, epoch, args, writer=writer
            )

        train_time = timer()
        if train_loss is np.nan or train_loss > 999:
            print("TERMINATING TRAINING DUE TO NAN LOSS")
            return
        # val
        test_loss, test_acc = run_batches(
                model, None, None, test_loader, False, 1, -1, args
            )
        test_time = timer()

        if args.do_malicious:
            mal_loss, mal_acc = run_batches(model, opt, lr_scheduler,
                mal_loader, False, 1, -1, args)
            epoch_stats['mal_loss'] = mal_loss
            epoch_stats['mal_acc'] = mal_acc
            if args.use_tensorboard:
                writer.add_scalar('Loss/mal',   mal_loss,         epoch)
                writer.add_scalar('Acc/mal',    mal_acc,          epoch)

        # report epoch results
        epoch_stats.update({
            'train_time': train_time,
            'train_loss': train_loss,
            'train_acc':  train_acc,
            'test_loss':  test_loss,
            'test_acc':   test_acc,
            'total_time': timer.total_time,
        })
        lr = lr_scheduler.get_last_lr()[0]
        summary = union({'epoch': epoch+1,
                         'lr': lr},
                        epoch_stats)
        for logger in loggers:
            logger.append(summary)
        if args.use_tensorboard:
            writer.add_scalar('Loss/train', train_loss,       epoch)
            writer.add_scalar('Loss/test',  test_loss,        epoch)
            writer.add_scalar('Acc/train',  train_acc,        epoch)
            writer.add_scalar('Acc/test',   test_acc,         epoch)
            writer.add_scalar('Time/train', train_time,       epoch)
            writer.add_scalar('Time/test',  test_time,        epoch)
            writer.add_scalar('Time/total', timer.total_time, epoch)
            writer.add_scalar('Lr',         lr,               epoch)

    return summary

#@profile
def run_batches(model, opt, lr_scheduler, loader,
                training, epoch_fraction, epoch_num, args, writer=None):
    if not training and epoch_fraction != 1:
        raise ValueError("Must do full epochs for val")
    if epoch_fraction > 1 or epoch_fraction <= 0:
        msg = "Invalid epoch_fraction {}.".format(epoch_fraction)
        msg += " Should satisfy 0 < epoch_fraction <= 1"
        raise ValueError(msg)

    model.train(training)
    losses = []
    accs = []

    start_time = 0
    mal_grad = 0
    if training:
        num_clients = loader.dataset.num_clients
        spe = steps_per_epoch(args.local_batch_size, loader.dataset,
                              args.num_workers)
        for i, batch in enumerate(loader):
            # only carry out an epoch_fraction portion of the epoch
            #if batch[0].size()[0] > 500:
            #    print("BATCH SIZE", batch[0].size())
            if i > spe * epoch_fraction:
                break
            batch.append(epoch_num * torch.ones_like(batch[0]))

            lr_scheduler.step()

            if lr_scheduler.get_last_lr()[0] == 0:
                # hack to get the starting LR right for fedavg
                print("HACK STEP")
                opt.step()

            if args.local_batch_size == -1:
                expected_num_clients = args.num_workers
                if torch.unique(batch[0]).numel() < expected_num_clients:
                    # skip if there weren't enough clients left
                    msg = "SKIPPING BATCH: NOT ENOUGH CLIENTS ({} < {})"
                    print(msg.format(torch.unique(batch[0]).numel(),
                                     expected_num_clients))
                    continue
            else:
                expected_numel = args.num_workers * args.local_batch_size
                if batch[0].numel() < expected_numel:
                    # skip incomplete batches
                    msg = "SKIPPING BATCH: NOT ENOUGH DATA ({} < {})"
                    print(msg.format(batch[0].numel(), expected_numel))
                    continue

            loss, acc = model(batch)
            #if args.robustagg in ["bulyan, krum"]:
            #print("IDX: {}, Loss: {:0.5f}, Acc: {:0.5f}".format(i, loss.mean().item(), acc.mean().item()))
            if np.any(np.isnan(loss)):
                print(f"LOSS OF {np.mean(loss)} IS NAN, TERMINATING TRAINING")
                return np.nan, np.nan

            weight_update = opt.step()
            #model.zero_grad()
            losses.extend(loss)
            accs.extend(acc)
            if args.dataset_name == "FEMNIST":
                lr = lr_scheduler.get_last_lr()[0]
                print("LR: {:0.5f}, Loss: {:0.5f}, Acc: {:0.5f}, Time: {:0.2f}".format(
                        lr, loss.mean().item(), acc.mean().item(), time.time() - start_time
                     ))
                start_time = time.time()
            if args.do_test:
                break
    else:
        for batch in loader:
            #if batch[0].numel() < args.valid_batch_size:
            #    print("SKIPPING VAL BATCH: TOO SMALL")
            #    continue
            loss, acc = model(batch)
            losses.extend(loss)
            accs.extend(acc)
            if args.do_test:
                break

    return np.mean(losses), np.mean(accs)

def get_data_loaders(args):
    train_transforms, val_transforms = {
     "ImageNet": (imagenet_train_transforms, imagenet_val_transforms),
     "CIFAR10": (cifar10_train_transforms, cifar10_test_transforms),
     "CIFAR100": (cifar100_train_transforms, cifar100_test_transforms),
     "FEMNIST": (femnist_train_transforms, femnist_test_transforms),
     "FashionMNIST": (fmnist_train_transforms, fmnist_test_transforms),
    }[args.dataset_name]

    dataset_class = globals()["Fed" + args.dataset_name]
    train_dataset = dataset_class(args, args.dataset_dir, args.dataset_name, transform=train_transforms,
                                  do_iid=args.do_iid, num_clients=args.num_clients,
                                  train=True, download=True)
    test_dataset = dataset_class(args, args.dataset_dir, args.dataset_name, transform=val_transforms,
                                 train=False, download=False)

    train_sampler = FedSampler(train_dataset,
                               args.num_workers,
                               args.local_batch_size)

    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_sampler,
                              num_workers=args.train_dataloader_workers)
                              #multiprocessing_context="spawn",
                              #pin_memory=True)
    test_batch_size = args.valid_batch_size * args.num_workers
    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=args.val_dataloader_workers)
                             #multiprocessing_context="spawn",
                             #pin_memory=True)
    print(len(train_loader), len(test_loader))

    mal_loader = None
    if args.do_malicious:
        mal_dataset = dataset_class(args, args.dataset_dir, args.dataset_name, transform=val_transforms,
                                 train=False, download=False, malicious=True)
        mal_loader = DataLoader(mal_dataset, 
                                #batch_size=test_batch_size,
                                shuffle=False,
                                batch_size=args.mal_targets,
                                num_workers=args.val_dataloader_workers
                                )

    return train_loader, test_loader, mal_loader

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    print("MY PID:", os.getpid())
    """
    import cProfile
    import sys
    # if check avoids hackery when not profiling
    # Optional; hackery *seems* to work fine even when not profiling,
    # it's just wasteful
    if sys.modules['__main__'].__file__ == cProfile.__file__:
        # Imports you again (does *not* use cache or execute as __main__)
        import cv_train
        # Replaces current contents with newly imported stuff
        globals().update(vars(cv_train))
        # Ensures pickle lookups on __main__ find matching version
        sys.modules['__main__'] = cv_train
    """

    # fixup
    #args = parse_args(default_lr=0.4)

    # fixup_resnet50
    #args = parse_args(default_lr=0.002)

    # fixupresnet9
    #args = parse_args(default_lr=0.06)

    args = parse_args()

    print(args)

    timer = Timer()
    args.lr_epoch = 0.0

    # reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # model class and config
    if args.do_test:
        model_config = {
            'channels': {'prep': 1, 'layer1': 1,
                         'layer2': 1, 'layer3': 1},
        }
        args.num_cols = 10
        args.num_rows = 1
        args.k = 10
    else:
        model_config = {
                'channels': {'prep': 64, 'layer1': 128,
                             'layer2': 256, 'layer3': 512},
        }
    if args.do_finetune:
        num_classes = num_classes_of_dataset(args.finetuned_from)
        num_new_classes = num_classes_of_dataset(args.dataset_name)
    else:
        num_classes = num_classes_of_dataset(args.dataset_name)
        num_new_classes = None

    model_config.update({"num_classes": num_classes,
                         "new_num_classes": num_new_classes})
    model_config.update({"bn_bias_freeze": args.do_finetune,
                         "bn_weight_freeze": args.do_finetune})
    if args.dataset_name == "FEMNIST":
        model_config["initial_channels"] = 1

    # comment out for Fixup
    model_config["do_batchnorm"] = args.do_batchnorm

    # make data loaders
    train_loader, test_loader, mal_loader = get_data_loaders(args)

    # instantiate ALL the things
    model_cls = getattr(models, args.model)
    model = model_cls(**model_config)

    if args.model[:5] == "Fixup":
        print("using fixup learning rates")
        params_bias = [p[1] for p in model.named_parameters()
                            if 'bias' in p[0]]
        params_scale = [p[1] for p in model.named_parameters()
                             if 'scale' in p[0]]
        params_other = [p[1] for p in model.named_parameters()
                             if not ('bias' in p[0] or 'scale' in p[0])]
        param_groups = [{"params": params_bias, "lr": 0.1},
                        {"params": params_scale, "lr": 0.1},
                        {"params": params_other, "lr": 1}]
    elif args.do_finetune:
        #PATH = args.checkpoint_path + args.model + str(args.mode) + str(args.do_dp) + str(do_malicious) + '.pt'
        PATH = args.checkpoint_path + args.model + str('fedavg') + str(True) + str(False) + '.pt'
        print("Finetuning from ", PATH)
        model.load_state_dict(torch.load(PATH))
        #for param in model.parameters():
        #    param.requires_grad = False
        #param_groups = model.finetune_parameters()
        param_groups = model.parameters()
    else:
        param_groups = model.parameters()
    opt = optim.SGD(param_groups, lr=1)

    model = FedModel(model, compute_loss_train, args, compute_loss_val, compute_loss_mal)
    opt = FedOptimizer(opt, args)

    # set up learning rate scheduler
    # original cifar10_fast repo uses [0, 5, 24] and [0, 0.4, 0]
    lr_schedule = PiecewiseLinear([0, args.pivot_epoch, args.num_epochs],
                                  [0, args.lr_scale,                  0])

    # grad_reduction only controls how gradients from different
    # workers are combined
    # so the lr is multiplied by num_workers for both mean and median
    spe = steps_per_epoch(args.local_batch_size,
                          train_loader.dataset,
                          args.num_workers)
    lambda_step = lambda step: lr_schedule(step / spe)
    lr_scheduler = LambdaLR(opt, lr_lambda=lambda_step)

    # set up output
    log_dir = make_logdir(args)
    if args.use_tensorboard:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None
    print('Finished initializing in {:.2f} seconds'.format(timer()))

    # and do the training
    train(model, opt, lr_scheduler, train_loader, test_loader, args,
          writer, loggers=(TableLogger(),), timer=timer, mal_loader=mal_loader)
    model.finalize()
    if args.do_checkpoint:
        print("Checkpointing model...")
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
        PATH = args.checkpoint_path + args.model + str(args.mode) + str(args.robustagg) + '.pt'
        torch.save(model.state_dict(), PATH)
        print("Model checkpointed at ", PATH)
        #loaded = torch.load(PATH)
        #model.load_state_dict(loaded)
