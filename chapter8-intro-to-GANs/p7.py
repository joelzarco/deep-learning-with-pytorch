#p7
# Training loop
# Implement and execute a GAN training loop
# The two optimizers, disc_opt and gen_opt, have been initialized as Adam() optimizers. The functions to compute the 
# losses that you defined earlier, gen_loss() and disc_loss(), are available to you. A dataloader is also prepared for you.

for epoch in range(1):
    for real in dataloader:
        cur_batch_size = len(real)
        
        disc_opt.zero_grad()
        # Calculate discriminator loss
        disc_loss = disc_loss(gen, disc, real, cur_batch_size, z_dim=16)
        # Compute gradients
        disc_loss.backward()
        disc_opt.step()

        gen_opt.zero_grad()
        # Calculate generator loss
        gen_loss = gen_loss(gen, disc, cur_batch_size, z_dim=16)
        # Compute generator gradients
        gen_loss.backward()
        gen_opt.step()

        print(f"Generator loss: {gen_loss}")
        print(f"Discriminator loss: {disc_loss}")
        break

# REcall that 
# 	- disc_loss()'s arguments are: gen, disc, real, cur_batch_size, z_dim
#	- gen_loss()'s arguments are: gen, disc, cur_batch_size, z_dim.