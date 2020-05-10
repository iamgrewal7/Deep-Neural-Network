# For importing the library from root folder
import sys
sys.path.insert(1, '../')

import numpy as np
import h5py
from iamg import NeuralNetwork
from utils import load_data

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


# Explore your dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))
print()

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.


layers_dims = [12288, 20, 7, 5, 1]
neural_network = NeuralNetwork(train_x, train_y, layer_dims = layers_dims, learning_rate = 0.0075, lambd=0.6)
neural_network.train(3000, print_cost=True)
neural_network.predict(train_x, train_y)


