# How to setup Keras in Anaconda with GPU support

Keras ( https://keras.io/ ) is a meta-framework / high-level API for neural networks. The current version still supports multiple backends like Tensorflow, Theano or CNTK, but in the future only Tensorflow will be supported. See section "test Keras".
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

#### test tensorflow GPU support

To test, if tensorflow can find and use your GPU, open python and type:

```python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
tf.test.gpu_device_name() 
```
expected result:
> Found device 0 with properties:
> pciBusID: 0000:01:00.0 name: GeForce RTX 2070 SUPER computeCapability: 7.5

#### test Keras

Keras is now the official high level API for tensorflow. From now on, if you want to use Keras, you should use the tensorflow keras submodule: "import tensorflow.keras".

at the anaconda prompt, open the python editor spyder: 

```
spyder
```

and paste this small python program into a new python document:

```python
from numpy import *
from tensorflow.keras import *

model = models.Sequential()

model.add(layers.Dense(units=64, activation='relu', input_dim=100))
model.add(layers.Dense(units=10, activation='softmax'))

model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
              metrics=['accuracy'])


# Generate dummy data

data = random.random((1000, 100))
labels = random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```
