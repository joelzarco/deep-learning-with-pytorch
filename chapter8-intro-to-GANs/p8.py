#P8
# Generating images
# Perform visual inspection to see if the generation resembles desired output.
# To do this, you will create random noise as input for the generator, pass it to the model and plot the outputs.

num_images_to_generate = 5
# Create random noise tensor
noise = torch.randn(num_images_to_generate, 16)

# Generate images
with torch.no_grad():
    fake = gen(noise)
print(f"Generated tensor shape: {fake.shape}")
    
for i in range(num_images_to_generate):
    # Slice fake to select i-th image
    image_tensor = fake[i, :, :, :]
    # Permute the image dimensions from (color, height, width) to (hight, width, color)
    image_tensor_permuted = image_tensor.permute(1, 2, 0)
    plt.imshow(image_tensor_permuted)
    plt.show()

# sample image saved to files