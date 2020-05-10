import numpy as np
from utils import relu, sigmoid, relu_backward, sigmoid_backward


class NeuralNetwork:
  def __init__(self, X_train, y_train, layer_dims, learning_rate, lambd):
    self.X = X_train
    self.Y = y_train
    self.layer_dims = layer_dims
    self.state = self.initilaize_parameters()
    self.learning_rate = learning_rate
    self.gradients = {}
    self.AL = None
    self.cost = None
    self.lambd = lambd
  
  def initilaize_parameters(self):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters 1: {"W1", "b1"}, ..., "WL", "bL":
                    W-lth -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    b-lth -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(1)
    return {
      layer: {
        'W': np.random.randn(self.layer_dims[layer], self.layer_dims[layer-1]) / np.sqrt(self.layer_dims[layer-1]),
        'b': np.zeros((self.layer_dims[layer], 1))
      } for layer in range(1, len(self.layer_dims))
    }


  def compute_cost(self, print_cost=True):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    """
    L2_regularization_cost = (self.lambd/(2*self.Y.shape[1])) * sum(np.sum(np.square(value['W'])) for _, value in self.state.items())
    self.cost = np.squeeze(
      (1./self.Y.shape[1]) * (-np.dot(self.Y,np.log(self.AL).T) - np.dot(1-self.Y, np.log(1-self.AL).T))
    ) + L2_regularization_cost
    assert(self.cost.shape == ())

  
  def forward_propagation(self):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    """
    A_prev = self.X
    L = len(self.state)

    for layer in range(1, L):
      """
      Z = W.A_prev + b
      A = func(z) 
      """
      # Inner layers use relu
      Z = np.dot(self.state[layer]['W'], A_prev) + self.state[layer]['b']
      A = relu(Z)
      self.state[layer]['A'] = A
      self.state[layer]['A_prev'] = A_prev
      self.state[layer]['Z'] = Z
      A_prev = A
    
    # Last layer use sigmoid
    ZL = np.dot(self.state[L]['W'], A_prev) + self.state[L]['b']
    AL = sigmoid(ZL)
    self.state[L]['A'] = AL
    self.state[L]['A_prev'] = A_prev
    self.state[L]['Z'] = ZL

    assert(AL.shape == (1, self.X.shape[1]))
    self.AL = AL

  
  def backward_propagation(self):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    """
    L = len(self.state)
    dAL = - (np.divide(self.Y, self.AL) - np.divide(1 - self.Y, 1 - self.AL))
    dZL = sigmoid_backward(dAL, self.state[L]['Z'])
    gradients ={
      L: {
        'dZ': dZL,
        'dA': dAL
      }
    }

    for layer in reversed(range(L)):
      A_prev = self.state[layer+1]['A_prev']
      W = self.state[layer+1]['W']
      b = self.state[layer+1]['b']
      m = A_prev.shape[1]
      dZ = gradients[layer+1]['dZ']

      dW = (1/m) * np.dot(dZ, A_prev.T) + (self.lambd/m)*W
      db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
      dA_prev = np.dot(W.T, dZ)

      assert (dA_prev.shape == A_prev.shape)
      assert (dW.shape == W.shape)
      assert (db.shape == b.shape)
      
      gradients[layer+1]['dW'] = dW
      gradients[layer+1]['db'] = db

      gradients[layer] = {'dA': dA_prev}
      if layer:
        gradients[layer]['dZ'] = relu_backward(dA_prev, self.state[layer]['Z'])
    
    self.gradients = gradients


  def update_state(self):
    """
    Update state using gradient descent
    """
    L = len(self.state)
    
    for layer in range(1, L+1):
      self.state[layer]['W'] -= self.learning_rate * self.gradients[layer]['dW']
      self.state[layer]['b'] -= self.learning_rate * self.gradients[layer]['db'] 
  

  def train(self, iterations, print_cost=True):
    """
    Train nerual network
    Arguments:
    iterations -- number of iterations
    """
    for i in range(iterations):
      self.forward_propagation()
      self.compute_cost()
      self.backward_propagation()
      self.update_state()
      if print_cost and i % 100 == 0:
        print(f"Cost after iteration {i}: {np.squeeze(self.cost)}")


  def predict(self, X_test, y_test):
    """
    Make prediction using trained neural network
    Arguments:
    X_test -- test dataset
    y_test -- Real y 
    """
    self.X = X_test
    self.forward_propagation()
    m = X_test.shape[1]
    
    p = np.zeros((1, m))
    for i in range(0, self.AL.shape[1]):
      if self.AL[0,i] > 0.5:
          p[0,i] = 1
      else:
          p[0,i] = 0
      
    print("Accuracy: "  + str(np.sum((p == y_test)/m)))



