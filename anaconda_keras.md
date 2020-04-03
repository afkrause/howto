# How to setup Keras in Anaconda with GPU support

Keras ( https://keras.io/ ) is a meta-framework / high-level API for neural networks. The current version still supports multiple backends like Tensorflow, Theano or CNTK, but in the future only Tensorflow will be supported.
To enable GPU support for NVIDIA GPU's , you need to install CUDA.


## install Anaconda

Anaconda is a preconfigured python environment for scientific computations. Using anaconda solves a lot of dependency issues.
Download anaconda with Python 3.7 and select the 64bit edition (important!)

https://www.anaconda.com/distribution/

Next, test anaconda. go to the start menu and select "Anaconda Prompt". enter *python* and test if everything works as expected.

### install Tensorflow usind conda package manager

The conda package manager is a tool that helps to deal with external library / dll dependencies, going beyond the capabilites of pip. For example, here, conda installs NVIDIA's *proprietary*, closed source the CUDA parallel computing framework and the cudnn sdk (to download that manually would require a totally annoying user registration and login at nvidias homepage).

First, you need to upgrade anaconda and conda to the latest and greatest:

```
conda update conda
conda update --all
```
This is very important, otherwise conda installs the old tensorflow 1.1x sdk instead of tensorflow 2.0

Next, install tensorflow and all dependencies:

```
conda install tensorflow-gpu
```

#### test GPU support

To test, if tensorflow can find and use your GPU, open python and type:

```python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
tf.test.gpu_device_name() 
```
expected result:
> Found device 0 with properties:
> pciBusID: 0000:01:00.0 name: GeForce RTX 2070 SUPER computeCapability: 7.5

### install Keras

at the anaconda prompt, enter: 

```
conda install keras
```

#### test Keras

at the anaconda prompt, open the python editor spyder: 

```
spyder
```

and paste this small python program into a new python document:

```python
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
              metrics=['accuracy'])


# Generate dummy data

data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```


### Manual installation using PIP package manager

Just in case something went wrong; you dont like conda or anaconda, you might try to setup tensorflow using the pip package manager. 
Now you need to manually install CUDA abd cudnn.

#### NVIDIA GPU: install CUDA and cudnn
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

##### TODO: cudnn
copy the dll into a directory that ins in the PATH. 

#### install tensorflow and keras with pip
TODO
