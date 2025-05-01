#p7
# Region proposal networks like Faster CNN
# this part defines just the achor sizes and ratios

# Import AnchorGenerator
from torchvision.models.detection.rpn import AnchorGenerator

# Configure anchor size
anchor_sizes = ((32, 64, 128),)

# Configure aspect ratio
aspect_ratios = ((0.5, 1.0, 2.0),)

# Instantiate AnchorGenerator
rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

# Part 2 is to define a Faster R-CNN model that can detect objects of different sizes in an image

# Import MultiScaleRoIAlign
from torchvision.ops import MultiScaleRoIAlign

# Instantiate RoI pooler (region of interest)
roi_pooler = MultiScaleRoIAlign(
	featmap_names=['0'],
	output_size=7,
	sampling_ratio=2,
)

mobilenet = torchvision.models.mobilenet_v2(weights="DEFAULT")
backbone = nn.Sequential(*list(mobilenet.features.children()))
backbone.out_channels = 1280

# Create Faster R-CNN model
model = FasterRCNN(
	backbone=backbone,
	num_classes=2,
	anchor_generator=anchor_generator,
	box_roi_pool=roi_pooler,
)