#p5
#Generator Loss
# Define a loss function for the generator

# Recall that the generator's job is to produce such fake images that would fool the discriminator into classifying them as real
# Therefore, the generator incurs a loss if the images it generated are classified by the discriminator as fake (label 0)

def gen_loss(gen, disc, criterion, num_images, z_dim):
    # Define random noise
    noise = torch.randn(num_images, z_dim)
    # Generate fake image
    fake = gen(noise)
    # Get discriminator's prediction on the fake image
    disc_pred = disc(fake)
    # Compute generator loss
    criterion = nn.BCEWithLogitsLoss()
    gen_loss = criterion(disc_pred, torch.ones_like(disc_pred))
    return gen_loss

