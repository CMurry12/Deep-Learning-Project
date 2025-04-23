"""
Description:
Manually implement a   Convolutional  Neural  Network (CNN) from scratch to 
classify handwritten digits from the MNIST dataset.

(1) Forward propagation function (including a 2D convolution operation). 
(2) Backward propagation function (including gradient calculations for convolution kernels and fully connected layer). 
(3) Train function (for parameter updates of CNN neural network by loss). 
(4) ReLU activation (for the 1st layer). 
(5) Softmax activation (for the 2nd layer). 
(6) Cross-entropy function (for calculating loss). 
(7) Main function (for training and testing designed CNN by train dataloader and test dataloader respectively).

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
# ReLU activation (for the 1st layer). 
def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

# Derivative of ReLU activation functions
def relu_derivative(x):
    """Derivative of ReLU used in backpropagation"""
    return (x > 0).astype(float)

# Softmax activation (for the 2nd layer)
def softmax(x):
    """Softmax activation for output layer (multi-class classification)"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Cross-entropy function (for calculating loss). 
def cross_entropy_loss(y_pred, y_true):
    """Cross-entropy loss for classification"""
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    return np.sum(log_likelihood) / m


# ===================== Data Loading ===================== #
def dataloader(train_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)


# ===================== CNN Structure ===================== #
class CNN:
    """
    Represent a Convolutional Neural Network (CNN) using deep learning techniques for 
    handwritten digits classification on MNIST dataset.
    """
    def __init__(self, input_size, num_filters, kernel_size, fc_output_size, lr):
        """
        Initialization.
        
        :param input_size: 
        :param num_filters: number of convolutional filters (kernels) to apply in the convolutional layer
        :param kernel_size: height and width of the square convolutional filters (e.g., 3 means a 3x3 filter)
        :param fc_output_size: number of output classes for classification.
        :param lr: learning rate used for updating weights during training via gradient descent.
        
        """
        # Initialize convolution kernel with Gaussian distribution
        self._kernel = np.random.randn(kernel_size, num_filters, kernel_size) * 0.01
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        
        # Compute output size after convolution (no padding)
        conv_output_size = input_size - kernel_size + 1
        
        # Flattened size after ReLU to feed into the fully connected layer
        self._fc_input_size = conv_output_size * conv_output_size * num_filters
        
        # Initialize weights and biases for fully connected layer
        self._W_fc = np.random.randn(self._fc_input_size, fc_output_size) * 0.01
        self._b_fc = np.zeros((1, fc_output_size))
        
        self._lr = lr # learning rate
        
        self._z1 = None # Pre-activation of layer 1
        self._a1 = None # Activation of layer 1 (ReLU)
        self._a1_flat = None # Flatten self._a1 for Fully Connected Layer
        self._z2 = None # Pre-activation of layer 2
        self._a2 = None # Output probabilities (via softmax)
    
    # Convolution layer
    def conv2d(self, x):
        """Description.
        
        :param x: Batch of MNIST images (shape [batch_size, 28, 28]).
        
        return result_of_conv_layer
        """
        batch_size, input_height, input_width = x.shape
        k = self._kernel_size
        output_height = input_height - k + 1
        output_width = input_width - k + 1
        result_of_conv_layer = np.zeros((batch_size, self._num_filters, output_height, output_width))
        
        # Slide the kernel across each spatial position
        for f in range(self._num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    region = x[:, i:i+k, j:j+k] # extract local patch
                    result_of_conv_layer[:, f, i, j] = np.sum(region * self._kernel[f], axis=(1, 2))
                
        return result_of_conv_layer
    
    # Forward pass through the CNN
    def forward(self, x):
        """
        Forward pass through the CNN.
        
        :param x: Batch of MNIST images (shape [batch_size, 28, 28]).
        
        return : Prediction probabilities after passing through the two layers.
        """
        # Layer 1
        self._z1 = self.conv2d(x) # convolution output
        self._a1 = relu(self._z1) # ReLU activation for layer 1
        
        self._a1_flat = self._a1.reshape(x.shape[0], -1) # Flatten self._a1 for Fully Connected Layer
        
        # Layer 2
        self._z2 = np.dot(self._a1_flat, self._W_fc) + self._b_fc # Fully Connected Layer
        self._a2 = softmax(self._z2) # Softmax activation → gives class probabilities
        
        return self._a2 # This returned value is 'pred'
    
    # Backward pass to compute gradients and update weights
    def backward(self, x, y, pred):
        """
        Computes gradients for convolution kernels weighs and fully connected weights/biases.
        Compute gradients averaged over the current batch (e.g., divide gradients by the batch size).
        
        :param x: Batch of images
        :param y: ground-truth labels
        :param pred: predictions from forward propagation.
        """
        m = y.shape[0] # Batch size
        
        # 1. one-hot encode the labels (Convert true labels to one-hot encoding)
        y_onehot = np.zeros_like(pred)
        y_onehot[np.arange(m), y] = 1
        
        # 2. Calculate softmax cross-entropy loss gradient (Gradient of loss w.r.t. z2 (output layer pre-activation))
        dz2 = (pred - y_onehot) / m
        dW_fc = np.dot(self._a1_flat.T, dz2)
        db_fc = np.sum(dz2, axis=0, keepdims=True)
        
        # 3. Calculate fully connected layer gradient
        da1_flat = np.dot(dz2, self._W_fc.T)
        da1 = da1_flat.reshape(self._a1.shape)
        
        # 4. Backpropagate through ReLU
        dz1 = da1 * relu_derivative(self._z1)
        
        # 5. Calculate convolution kernel gradient
        dK = np.zeros_like(self._kernel)
        for f in range(self._num_filters):
            for i in range(self._kernel.shape[1]):
                for j in range(self._kernel.shape[2]):
                    region = x[:, i:i + dz1.shape[2], j:j + dz1.shape[3]]
                    dK[f, i, j] = np.sum(region * dz1[:, f])
                
        # 6. Update parameters (Update the weights and biases using gradient descent (θ_new = θ_prev - ⍺ * ∇f))
        self._W_fc = self._W_fc - self._lr * dW_fc
        self._b_fc = self._b_fc - self._lr * db_fc
        self._kernel = self._kernel - self._lr * dK
        
    
    def train(self, x, y):
        """
        Uses batch-based training (i.e., process data in batches of size 128 as provided by the dataloader)
        Computes forward function, uses the result(pred: predection) to compute the cross-entropy loss, 
        then computes the backward function.
        
        :param x: batch of images
        :param y: ground-truth labels
        
        retun loss: cross-entropy loss
        """
        # call forward function
        pred = self.forward(x)
        # calculate loss
        loss = cross_entropy_loss(pred, y)
        # call backward function
        self.backward(x, y, pred)
        
        return loss


# ===================== Training Process ===================== #
def main():
    # First, load data
    train_loader, test_loader = load_data()
    
    # Second, define hyperparameters
    input_size = 28
    num_filters = 1
    kernel_size = 3
    fc_output_size = 10
    learning_rate = 0.01
    num_epochs = 5
    
    # Initialize model
    model = CNN(input_size, num_filters, kernel_size, fc_output_size, learning_rate)
    
    # Then, train the model
    for epoch in range(num_epochs):
        total_loss = 0
        
        for inputs, labels in train_loader: # define training phase for training model
            x = inputs.numpy().squeeze(1)
            y = labels.numpy()
            loss = model.train(x, y)
            total_loss += loss

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}") # print the loss for each epoch

    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.numpy().squeeze(1) # Flatten and convert to NumPy
        y = labels.numpy()
        pred = model.forward(x) # the model refers to the model that was trained during the training phase
        predicted_labels = np.argmax(pred, axis=1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(y)
    print(f"Test Accuracy: {correct_pred / total_pred:.4f}")

if __name__ == "__main__":
    main()
