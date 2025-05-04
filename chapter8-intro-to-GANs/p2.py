#p2
# Build a discriminator that takes generator's output as input and produces a binary prediction

class Discriminator(nn.Module):
    def __init__(self, im_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            disc_block(im_dim, 1024),
            disc_block(1024, 512),
            # Define last discriminator block
            disc_block(512, 256),
            # Add a linear layer
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # Define the forward method
        return self.disc(x)

'''
disc_block is defined as:

def disc_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LeakyReLU(0.2)
    )
'''
