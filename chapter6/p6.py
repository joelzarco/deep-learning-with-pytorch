#p6.py
# Classifier block
# Your next task is to create a classifier block that will replace the original VGG16 classifier. 
# You decide to use a block with two fully connected layers with a ReLU activation in between
# The vgg_model and input_dim you defined in the last exercise are available in your workspace, 
# and torch and torchvision.models have been imported.

# Create a variable with the number of classes
num_classes = 2
    
# Create a sequential block
classifier = nn.Sequential(
	# Create a linear layer with input features
	nn.Linear(input_dim, 512),
	nn.ReLU(),
	# Add the output dimension to the classifier
	nn.Linear(512, num_classes),
)

# part2 Box regressor block
# Your final task is to create a regressor block to predict bounding box coordinates

# Define the number of coordinates
num_coordinates = 4

bb = nn.Sequential(  
	# Add input and output dimensions
	nn.Linear(input_dim, 32),
	nn.ReLU(),
	# Add the output for the last regression layer
	nn.Linear(32, num_coordinates),
)