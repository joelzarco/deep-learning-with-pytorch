#p6
# test previous model to perform semantic segmentation on a car image

# Load model
model = UNet()
model.eval()

# Load and transform image
image = Image.open('car.jpg')
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0)

# Predict segmentation mask
with torch.no_grad():
    prediction = model(image_tensor).squeeze(0)

# Display mask
plt.imshow(prediction[1, :, :])
plt.show()

# result is good, saved in files :)