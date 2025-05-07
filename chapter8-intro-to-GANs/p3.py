#p3
# define a convolutional generator

# dc_gen_block is defined as follows:

def dc_gen_block(in_dim, out_dim, kernel_size, stride):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride),
        nn.BatchNorm2d(out_dim),
        nn.ReLU()
    )

class DCGenerator(nn.Module):
    def __init__(self, in_dim, kernel_size=4, stride=2):
        super(DCGenerator, self).__init__()
        self.in_dim = in_dim
        self.gen = nn.Sequential(
            dc_gen_block(in_dim, 1024, kernel_size, stride),
            dc_gen_block(1024, 512, kernel_size, stride),
            # Add last generator block
            dc_gen_block(512, 256, kernel_size, stride),
            # Add transposed convolution
            nn.ConvTranspose2d(256, 3, kernel_size, stride=stride),
            # Add tanh activation
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(len(x), self.in_dim, 1, 1)
        return self.gen(x)