#p1
# Image tensors

# Convert bbox into tensors
bbox_tensor = torch.tensor([bbox])

# Add a new batch dimension
bbox_tensor = bbox_tensor.unsqueeze(0)

# Resize image and transform tensor
transform = transforms.Compose([
  transforms.Resize(224),
  transforms.PILToTensor()
])

# Apply transform to image
image_tensor = transform(image)
print(image_tensor)

# we use transforms.ToTensor() float transformation for images to scale them in the range [0, 1]. 
# However, the bounding box requires an unscaled image type with the range [0, 255], so we use transforms.PILToTensor()