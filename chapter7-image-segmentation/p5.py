#p5
# Buld forward method for U-net model
# The goal of the decoder is to upsample the feature maps so that its output is of the same height and width as the U-Net's 
# input image. This will allow you to obtain pixel-level semantic masks

def forward(self, x):
    x1 = self.enc1(x)
    x2 = self.enc2(self.pool(x1))
    x3 = self.enc3(self.pool(x2))
    x4 = self.enc4(self.pool(x3))

    x = self.upconv3(x4)
    x = torch.cat([x, x3], dim=1)
    x = self.dec1(x)

    x = self.upconv2(x)
    x = torch.cat([x, x2], dim=1)
    x = self.dec2(x)

    # Define the last decoder block with skip connections
    x = self.upconv1(x)
    x = torch.cat([x, x1], dim=1)
    x = self.dec3(x)

    return self.out(x)

    