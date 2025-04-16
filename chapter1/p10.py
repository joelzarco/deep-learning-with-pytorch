#p10
# convolution layers and max pooling
# input images are 64x64
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Define feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        # Define classifier
        self.classifier = nn.Linear(64*16*16, num_classes)
    
    def forward(self, x):  
        # Pass input through feature extractor and classifier
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# layer by layer
'''
Conv2d(3,32) with padding=1: Output: 32 channels × 64 × 64
MaxPool2d(2): Output: 32 × 32 × 32
Conv2d(32,64) with padding=1: Output: 64 × 32 × 32
MaxPool2d(2): Output: 64 × 16 × 16
nn.Flatten(): Output: 64 × 16 × 16 = 16,384 features
The linear layer nn.Linear(64*16*16, num_classes) needs to match this exact flattened dimension (16,384) as its input size
'''

# or use this trick
#self.classifier = nn.Linear(features.shape[1], num_classes)