#p3
# Predicting bounding boxes
# The model is available as model and it's already in the evaluation mode. The test_image is also available, 
# and torch and torch.nn as nn have been imported

# Get model's prediction
with torch.no_grad():
    output = model(test_image)

# Extract boxes from the output
boxes = output[0]['boxes']

# Extract scores from the output
scores = output[0]['scores']

print(boxes, scores)


'''
tensor([[0.0459, 0.4543, 0.6344, 0.6124],
            [0.9197, 0.0498, 0.4779, 0.8986],
            [0.8335, 0.1734, 0.4905, 0.1828],
            [0.7738, 0.3889, 0.8853, 0.4540],
            [0.4424, 0.6377, 0.3331, 0.2442]]) 
tensor([0.9480, 0.7986, 0.9829, 0.7108, 0.7335])
'''