# How to setup Keras in Anaconda with GPU support

Keras ( https://keras.io/ ) is a meta-framework / high-level API for neural networks. The current version still supports multiple backends like Tensorflow, Theano or CNTK, but in the future only Tensorflow will be supported.
To enable GPU support for NVIDIA GPU's , you need to install CUDA.

## NVIDIA GPU: install CUDA
The Tensorflow backend provides both GPU and CPU acceleration. For GPU acceleration to work, you need to install NVIDIA's *proprietary*, closed source the CUDA parallel computing framework. 
Unfortunately, CUDA currently dominates to market. 
Tensorflow requres a specific CUDA version. Please check the exact version required before downloading CUDA.
Go to the tensorflow homepage:

https://www.tensorflow.org/install/gpu

and check section *Software requirements*.
The current, stable Tensorflow version requires CUDA 10.1 .

Go to the CUDA toolkit archive:

https://developer.nvidia.com/cuda-toolkit-archive

and download the required CUDA version.

## install Anaconda

Anaconda is a preconfigured python environment for scientific computations. Using anaconda solves a lot of dependency issues.
Download anaconda with Python 3.7 and select the 64bit edition (important!)

https://www.anaconda.com/distribution/

## install Tensorflow within Anaconda

