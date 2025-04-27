#p2
# CNN model for multiclass classification
# assume image are initially 64x64

class MultiClassImageClassifier(nn.Module):
  
    # Define the init method
    def __init__(self, num_classes):
        super(MultiClassImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Create a fully connected layer
        self.fc = nn.Linear(16*32*32, num_classes)
        
        # Create an activation function
        self.softmax = nn.Softmax(dim=1) # Softmax: Converts class scores into probabilities summing to 1 for each sample
        # dim=1: Targets the class dimension (columns), ensuring Each row (sample) becomes a probability distribution
        # where each row sums to 1