#p3
# Segmenting with pre-trained mask R-CNN
# use pre-trained Mask R-CNN model to perform instance segmentation on two cats image
# the model has been pretrained using COCO dataset

# Image from PIL, torch, transforms from torchvision, and maskrcnn_resnet50_fpn have been imported


# Load a pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load an image and convert to a tensor
image = Image.open("two_cats.jpg")
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0)

# Perform inference
with torch.no_grad():
    prediction = model(image_tensor)
    print(prediction)

# coco dataset https://cocodataset.org/#overview

# P2 Displaying Soft masks

# Extract masks and labels from prediction
masks = prediction[0]['masks']
labels = prediction[0]['labels']

# Plot image with two overlaid masks
for i in range(2):
    plt.imshow(image)
    # Overlay the i-th mask on top of the image
    plt.imshow(masks[i,0], cmap='jet', alpha=0.5)
    plt.title(f"Object: {class_names[labels[i]]}")
    plt.show()