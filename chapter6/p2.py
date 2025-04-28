#p2
# draw bounding box oer object

# torch, torchvision,torchvision.transforms have been imported. Image has been already transformed to tensors as image_tensor. 
# The coordinates have been assigned to the variables: x_min, y_min, x_max, y_max

# Import draw_bounding_boxes
from torchvision.utils import draw_bounding_boxes

# Define the bounding box coordinates
bbox = [x_min, y_min, x_max, y_max]
bbox_tensor = torch.tensor(bbox).unsqueeze(0)

# Implement draw_bounding_boxes
img_bbox = draw_bounding_boxes(image_tensor, bbox_tensor, width=3, colors="red")

# Tranform tensors to image
transform = transforms.Compose([
    transforms.ToPILImage()
])
plt.imshow(transform(img_bbox))
plt.show()

# shows red rect over espresso coffee