### READ THIS FIRST

This repo is an anonymized version of an existing repository of GitHub, for the NeurIPS 2021 submission: SparseFed: Provably Defending Against Backdoor Attacks in Federated Learning with Sparsification. So if you see another repo that looks identical to this, we are not stealing anyone's code, that's my repo.

# SparseFed

This repo contains an implementation of model poisoning attacks on a federated learning system.

It comes with a few experimental setups; various Residual Networks on CIFAR10, CIFAR100, FEMNIST, ImageNet (`cv_train.py`) and GPT2 on PersonaChat (`gpt2_train.py`) (attack is currently not implemented for PersonaChat)

There are a variety of command-line args which are best examined by looking at `utils.py`

The server is contained in `fed_aggregator.py` and the worker is contained in `fed_worker.py`

To use sketching, you need to install https://github.com/nikitaivkin/csh
