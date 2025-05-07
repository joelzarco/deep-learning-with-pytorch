#P9
#  Evaluate your GAN using the Fréchet Inception Distance

# Import FrechetInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

# Instantiate FID
fid = FrechetInceptionDistance(feature=64)

# Update FID with real images
fid.update((fake * 255).to(torch.uint8), real=False)
fid.update((real * 255).to(torch.uint8), real=True)

# Compute the metric
fid_score = fid.compute()
print(fid_score)

# tensor(7.5159)
# The FID below 10 indicates a really good quality and variety of generated images!