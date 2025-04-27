#p6
# Load and set weights from a pre-trained model


# Import resnet18 model
from torchvision.models import resnet18, ResNet18_Weights

# Initialize model with default weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Set model to evaluation mode
model.eval()

# Initialize the transforms
transform = weights.transforms()