from setuptools import setup

setup(name='CommEfficient',
      version='0.0.1',
      description='Communication Efficient SGD',
      keywords='communication efficient sketch sgd',
      url='https://github.com/sparsefed/sparsefed.git',
      author='SparseFed',
      author_email='thisemail@doesnot.exist',
      license='GNU GPL-3.0',
      packages=['CommEfficient'],
      # install_requires=[
      #     'csvec',
      # ],
      include_package_data=True,
      zip_safe=False)
