#p1
# Define a generator

class Generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Generator, self).__init__()
        # Define generator block
        self.generator = nn.Sequential(
            gen_block(in_dim, 256),
            gen_block(256, 512),
            gen_block(512, 1024),
          	# Add linear layer
            nn.Linear(1024, out_dim),
            # Add activation
            nn.Sigmoid()
        )

    def forward(self, x):
      	# Pass input through generator
        return self.generator(x)

'''
gen_block is defined as:
def gen_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True)
    )

'''