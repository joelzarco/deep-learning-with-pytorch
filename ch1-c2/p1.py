#p1
# CNN model for binary classification
# Create a convolutional layer with 3 channels, 16 output channels, kernel size of 3, stride of 1, and padding of 1
# Create a fully connected layer with an input size of 16x32x32 and a number of classes equal to 1;
# Create a sigmoid activation function


class BinaryImageClassifier(nn.Module):
    def __init__(self):
        super(BinaryImageClassifier, self).__init__()
        
        # Create a convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        # Create a fully connected layer
        self.fc = nn.Linear(16*32*32, 1)
        
        # Create an activation function
        self.sigmoid = nn.Sigmoid(x)