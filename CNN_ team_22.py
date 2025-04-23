import numpy as np

# Class for CNN
class SimpleCNN:
    # initates the kernal size and learning rate
    def __init__(self, kernel_size=3, learning_rate=0.01):
        self.kernel_size = kernel_size
        self.lr = learning_rate
        self.init_parameters()

    # sets up random starting parms
    def init_parameters(self):
        self.kernel = np.random.randn(self.kernel_size, self.kernel_size) * 0.01  # Conv kernel
        self.fc_weights = np.random.randn(10, (28 - self.kernel_size + 1)**2) * 0.01  # FC weights
        self.fc_biases = np.zeros(10)  # Bias for 10 output classes

    # runs relu
    def relu(self, x):
        return np.maximum(0, x)

    # runs softmax
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    # does cross entropy calculations
    def cross_entropy(self, preds, labels):
        batch_size = preds.shape[0]
        return -np.sum(np.log(preds[range(batch_size), labels])) / batch_size

    # deals with forward propagation
    # inputs: image batch of shape (size,28,28) = x
    # output: Output probabilities after softmax, shape is (size,10)
    def forward(self, x):
        # grab size
        batch_size = x.shape[0]
        conv_output = np.zeros((batch_size, 28 - self.kernel_size + 1, 28 - self.kernel_size + 1))
        
        # Convolution operation
        for i in range(batch_size):
            for row in range(28 - self.kernel_size + 1):
                for col in range(28 - self.kernel_size + 1):
                    region = x[i, row:row + self.kernel_size, col:col + self.kernel_size]
                    conv_output[i, row, col] = np.sum(region * self.kernel)
        
        # ReLU activation function
        relu_output = self.relu(conv_output)
        
        # Flatten the output to 1d vector
        flattened = relu_output.reshape(batch_size, -1)
        
        # adjust for weights and biases
        fc_output = np.dot(flattened, self.fc_weights.T) + self.fc_biases
        
        # Softmax activation function
        probs = self.softmax(fc_output)

        self.x = x  # Save input for backprop
        self.conv_output = conv_output
        self.relu_output = relu_output
        self.flattened = flattened
        self.probs = probs

        
        return probs

    # Deals with back propagation
    # input is labels = y, predictions = pred
    # Output adjusts weights
    def backward(self, y):
        # get shape of labels
        batch_size = y.shape[0]

        # Gradient of cross-entropy loss with softmax output
        dL_dfc_output = self.probs.copy()
        dL_dfc_output[range(batch_size), y] -= 1
        dL_dfc_output /= batch_size  # average over batch

        # Gradients for fully connected layer
        dL_dW_fc = np.dot(dL_dfc_output.T, self.flattened)  # shape: [10, flattened_dim]
        dL_db_fc = np.sum(dL_dfc_output, axis=0)  # shape: [10]

        # Gradient w.r.t. flattened input (backprop through FC)
        dL_dflattened = np.dot(dL_dfc_output, self.fc_weights)  # shape: [batch_size, flattened_dim]

        # Gradient of ReLU 
        dL_drelu = dL_dflattened.reshape(self.relu_output.shape)
        d_relu_dconv = self.conv_output > 0
        dL_dconv = dL_drelu * d_relu_dconv  # element-wise multiply

        # Gradient for convolution kernel
        dL_dkernel = np.zeros_like(self.kernel)
        for i in range(batch_size):
            for row in range(dL_dconv.shape[1]):
                for col in range(dL_dconv.shape[2]):
                    region = self.x[i, row:row+self.kernel_size, col:col+self.kernel_size]
                    dL_dkernel += dL_dconv[i, row, col] * region
        
        dL_dkernel /= batch_size  # average over batch

        # Update parameters
        self.fc_weights -= self.lr * dL_dW_fc
        self.fc_biases  -= self.lr * dL_db_fc
        self.kernel     -= self.lr * dL_dkernel

    #  Deals with training the model
    # inputs: train_loader = training data, test_loader = test info, and epoch = number of goes
    def train(self, train_loader, test_loader, epochs=5):
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            # go through a training cycle
            for i, (images, labels) in enumerate(train_loader):
                # Forward pass
                probs = self.forward(images)

                # Compute loss
                loss = self.cross_entropy(probs, labels)
                total_loss += loss

                # Predictions & Accuracy
                preds = np.argmax(probs, axis=1)
                correct += np.sum(preds == labels)
                total += labels.shape[0]

                # Backward pass
                self.backward(labels)

                # Print every 50 batches
                if (i + 1) % 50 == 0:
                    acc = correct / total * 100
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}], "
                        f"Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
            
            # Epoch summary
            train_acc = correct / total * 100
            avg_loss = total_loss / (i + 1)
            avg_loss = total_loss / (i + 1)
            print(f"Epoch [{epoch+1}] completed. Avg Loss: {avg_loss:.4f}, Training Accuracy: {train_acc:.2f}%")

            # Evaluate on test set
            test_acc = self.evaluate(test_loader)
            print(f"Test Accuracy after epoch {epoch+1}: {test_acc:.2f}%\n")


    # a function to evalute Accuracy
    def evaluate(self, test_loader):
        correct = 0
        total = 0
        for images, labels in test_loader:
            probs = self.forward(images)
            preds = np.argmax(probs, axis=1)
            correct += np.sum(preds == labels)
            total += labels.shape[0]
        return correct / total * 100

    # Deals with testing the model
    def test(self, test_loader):
        acc = self.evaluate(test_loader)
        print(f"Final Test Accuracy: {acc:.2f}%")

#-----------------------------------------------------------
# Oddly enough, this is just the data set
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transform to tensor and normalize between 0 and 1
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0,1]
])

# Download MNIST
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Convert torch tensors to NumPy arrays in batches
def numpy_data_loader(dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for images, labels in loader:
        # Convert shape from [B, 1, 28, 28] -> [B, 28, 28]
        np_images = images.squeeze(1).numpy()
        np_labels = labels.numpy()
        yield np_images, np_labels

# Create loaders for training and testing
train_loader = numpy_data_loader(train_dataset, batch_size=64)
test_loader  = numpy_data_loader(test_dataset, batch_size=64)

#-------------------------------------------------------------------

cnn = SimpleCNN(kernel_size=3, learning_rate=0.01)
cnn.train(train_loader, test_loader, epochs=5)
cnn.test(test_loader)
