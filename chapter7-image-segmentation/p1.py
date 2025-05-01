#p1
# In this and the next exercise, you will use the corresponding mask to segment the cat out of the image.
# First, you will need to load the mask and binarize it

# Load mask image
mask = Image.open('annotations/Egyptian_Mau_123.png')

# Transform mask to tensor
transform = transforms.Compose([transforms.ToTensor()])
mask_tensor = transform(mask)

# Create binary mask
binary_mask = torch.where(
    mask_tensor == 1/255, 
    torch.tensor(1.0),
    torch.tensor(0.0),
)

# Print unique mask values
print(binary_mask.unique())




# It's important to remember that the toTensor() transformation normalizes the pixel values by dividing them by 255