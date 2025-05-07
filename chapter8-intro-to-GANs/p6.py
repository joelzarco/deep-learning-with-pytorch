#p6
# Discriminator Loss
# Recall that the discriminator's job is to classify images either real or fake. 
# Therefore, the generator incurs a loss if it classifies generator's outputs as real (label 1) or the real images as fake (label 0).

def disc_loss(gen, disc, real, num_images, z_dim):
    criterion = nn.BCEWithLogitsLoss()
    noise = torch.randn(num_images, z_dim)
    fake = gen(noise)
    # Get discriminator's predictions for fake images
    disc_pred_fake = disc(fake)
    # Calculate the fake loss component
    fake_loss = criterion(disc_pred_fake, torch.zeros_like(disc_pred_fake))
    # Get discriminator's predictions for real images
    disc_pred_real = disc(real)
    # Calculate the real loss component
    real_loss = criterion(disc_pred_real, torch.ones_like(disc_pred_real))
    disc_loss = (real_loss + fake_loss) / 2
    return disc_loss


# The discriminator is rewarded it makes correct classifications: real images as real, and fake ones as fake