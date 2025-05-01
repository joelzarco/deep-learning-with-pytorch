#p2
# With the binary mask ready, you can use it to segment the object, that is the cat, out of the image
# transforms from torchvision have been imported, and the binary_mask you created in the previous exercise is available to you.

# Load image and transform to tensor
image = Image.open("images/Egyptian_Mau_123.jpg")
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image)

# Segment object out of the image
object_tensor = image_tensor * binary_mask

# Convert segmented object to image and display
to_pil_image = transforms.ToPILImage() # this is weird!
object_image = to_pil_image(object_tensor)
plt.imshow(object_image)
plt.show()

# cat's been taken out of the image, only the countour remains