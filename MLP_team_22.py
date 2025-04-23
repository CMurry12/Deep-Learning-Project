"""
Description:
Manually implement a Multi-Layer Perceptron (MLP) from scratch to 
classify handwritten digits from the MNIST dataset.

(1) Forward propagation function.
(2) Backward propagation function (including gradient calculations).
(3) Train function (for parameter updates of MLP neural network by loss).
(4) Sigmoid activation function (for the 1st layer).
(5) Softmax activation function (for the 2nd layer).
(6) Cross-entropy function (for calculating loss).
(7) Main function (for training and testing designed MLP by train dataloader and test dataloader
respectively).

Authors: [Abdoul Djalil Guyzmo Sawadogo, Chase Murry, Toan Le]
Date: Apr 27, 2025
"""

# CLEAN THE CONSOLE.
from os import system
system('clear')



import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ===================== Utility Functions ===================== #
# Activation functions (hidden layers)
def sigmoid(x):
    """Sigmoid activation for the hidden layer"""
    return 1 / (1 + np.exp(-x))

# Derivative of activation functions
def sigmoid_derivative(z):
    """Derivative of sigmoid used in backpropagation"""
    p = sigmoid(z)
    return p - p**2 # p - p**2 = p * (1 - p)

# Activation functions (output layers)
def softmax(x):
    """Softmax function for multi-class classification"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # stability trick
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Cross-entropy loss for classification
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]  # Batch size
    log_likelihood = -np.log(y_pred[range(m), y_true])  # Select log-prob of correct class
    loss = np.sum(log_likelihood) / m
    return loss


# ===================== Data Loading ===================== #
def dataloader(train_dataset, test_dataset, batch_size=128):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)


# ===================== MLP Structure ===================== #
class MLP:
    """
    Represent a Multi-Layer Perceptron (MLP) using deep learning techniques for 
    handwritten digits classification on MNIST dataset.
    """
    
    def __init__(self, input_size, hidden_size, output_size, lr):
        """
        Initialization.
        
        :param input_size:
        :param hidden_size:
        :param output_size:
        :param lr:
        """
        # Initialize weights with Xavier/He-style initialization for stability
        self._W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        self._b1 = np.zeros((1, hidden_size))  # Bias for hidden layer

        self._W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self._b2 = np.zeros((1, output_size))  # Bias for output layer
        
        self._lr = lr # Learning rate
        
        self._z1 = None # Pre-activation of hidden layer
        self._a1 = None # Activation of hidden layer (sigmoid)
        self._z2 = None # Pre-activation of output layer
        self._a2 = None # Output probabilities (via softmax)
        
    
    # forward propagation to get predictions
    def forward(self, x):
        """
        :param x: Batch of flattened MNIST images (shape [batch_size, 784]). (input size 28x28=784)
        
        return self._a2: Prediction probabilities after passing through the two layers.
        """
        # Layer 1
        self._z1 = np.dot(x, self._W1) + self._b1 #  Linear transformation for hidden layer: z1 = x•w1 + b1
        self._a1 = sigmoid(self._z1) # sigmoid activation for hidden layer: a1 = σ(z1)
        
        # Layer 2
        self._z2 = np.dot(self._a1, self._W2) + self._b2 # Linear transformation for output layer: z2 = a1•W2 + b2
        self._a2 = softmax(self._z2) # Softmax activation → gives class probabilities: a2 = sm(z2)
        
        return self._a2 # This returned value is 'pred'
    
    def backward(self, x, y, pred):
        """
        The function computes gradients averaged over the current batch (e.g., divide gradients by the batch size).
        Uses the chain rule to manually derive gradients for both layers.
        
        :param x: Batch of images
        :param y: ground-truth labels
        :param pred: predictions from forward propagation.
        """
        m = y.shape[0] # Batch size
        
        # Convert true labels to one-hot encoding
        y_onehot = np.zeros_like(pred)
        y_onehot[np.arange(m), y] = 1  # one-hot encoding

        # Compute the gradients:
        # Gradient of loss w.r.t. z2 (output layer pre-activation)
        dz2 = pred - y_onehot
        dW2 = np.dot(self._a1.T, dz2) / m  # Gradient w.r.t. W2
        db2 = np.sum(dz2, axis=0, keepdims=True) / m  # Gradient w.r.t. b2
        
        # Backprop into hidden layer
        da1 = np.dot(dz2, self._W2.T)  # Gradient flowing back into a1
        dz1 = da1 * sigmoid_derivative(self._z1)  # Elementwise multiply with sigmoid
        dW1 = np.dot(x.T, dz1) / m  # Gradient w.r.t. W1
        db1 = np.sum(dz1, axis=0, keepdims=True) / m  # Gradient w.r.t. b1
        
        # Update the weights and biases: Gradient descent (θ_new = θ_prev - ⍺*∇f)
        self._W1 = self._W1 - self._lr * dW1
        self._b1 = self._b1 - self._lr * db1
        self._W2 = self._W2 - self._lr * dW2
        self._b2 = self._b2 - self._lr * db2
    
    # Training step for one batch
    def train(self, x, y):
        """
        This function uses batch-based training (i.e., process data in batches of size 128 as provided by the dataloader)
        Computes forward function, uses the result(pred: predection) to compute the cross-entropy loss, 
        then computes the backward function.
        
        :param x: batch of images
        :param y: ground-truth labels
        
        retun loss: cross-entropy loss
        """
        # call forward function
        pred = self.forward(x)
        # calculate loss
        loss = cross_entropy_loss(y_pred=pred, y_true=y) # cross-entropy loss
        # call backward function
        self.backward(x, y, pred)
        
        return loss


# ===================== Training Process ===================== #
def main():
    # First, load data
    train_loader, test_loader = load_data()

    # Second, define hyperparameters
    input_size = 28 * 28 # MNIST images are 28x28 pixels
    hidden_size = 128
    output_size = 10
    learning_rate = 0.01
    num_epochs = 100
    
    
    model = MLP(input_size, hidden_size, output_size, learning_rate) # Initialize model


    # Then, train the model
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, labels in train_loader:  # define training phase for training model
            x = inputs.view(-1, input_size).numpy()  # Flatten and convert to NumPy
            y = labels.numpy()
            loss = model.train(x, y)
            total_loss += loss

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}") # print the loss for each epoch

    # Finally, evaluate the model
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.view(-1, input_size).numpy()
        y = labels.numpy()
        pred = model.forward(x)  # the model refers to the model that was trained during the raining phase
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred}")

if __name__ == "__main__":  # Program entry
    main()  
