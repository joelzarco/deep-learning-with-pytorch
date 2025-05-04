#p7
# Generate Semantic mask as first step to create a panoptic mask
# Instantiate the model
model = UNet()

# Produce semantic masks for the input image
with torch.no_grad():
    semantic_masks = model(image_tensor)

# Choose highest-probability class for each pixel
semantic_mask = torch.argmax(semantic_masks, dim=1)

# Display the mask
plt.imshow(semantic_mask.squeeze(0))
plt.axis("off")
plt.show()

# Semantic mask saved to files

# P2 Overlaying instance masks
# you can overwrite it with instance masks in the locations where the objects have been identified by the instance segmentation model
# Use pre-trained MaskRCNN model to produce instance segmentation masks
# loop over them and overlay the parts where an object is detected with high certainty on top of the semantic masks

# Instantiate model and produce instance masks
model = MaskRCNN()
with torch.no_grad():
    instance_masks = model(image_tensor)[0]["masks"]

# Initialize panoptic mask as semantic_mask
panoptic_mask = torch.clone(semantic_mask)

# Iterate over instance masks
instance_id = 3
for mask in instance_masks:
    # Set panoptic mask to instance_id where mask > 0.5
    panoptic_mask[mask > 0.5] = instance_id
    instance_id += 1
    
# Display panoptic mask
plt.imshow(panoptic_mask.squeeze(0))
plt.axis("off")
plt.show() # Image saved to files