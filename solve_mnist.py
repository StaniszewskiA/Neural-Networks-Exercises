import numpy as np

from network import Network
from fully_connected_layer import FullyConnectedLayer
from activation_layer import ActivationLayer
from activation import tanh, tanh_prime
from loss import mse, mse_prime

from keras.datasets import mnist
#from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical

#Load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Training data: 60000 samples
#Reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
#Encode output which is a number in range [0,9] into a vactor of size 10
#e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train)

#Same for test data: 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)

#Network
net = Network()
net.add(FullyConnectedLayer(28*28, 100))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FullyConnectedLayer(100, 50))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FullyConnectedLayer(50 ,10))
net.add(ActivationLayer(tanh, tanh_prime))

#Train on 1000 samples
net.use(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)

#Test on 3 samples
out = net.predict((x_test[0:3]))
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])