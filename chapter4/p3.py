#p3
# Two output Dataset and dataLoader
# In this and the following exercises, you will build a two-output model to predict both the character and 
# the alphabet it comes from based on the character's image

# Create dataset_train
dataset_train = OmniglotDataset(
    transform=transforms.Compose([
        transforms.ToTensor(),
      	transforms.Resize((64, 64)),
    ]),
    samples=samples,
)

# Create dataloader_train
dataloader_train = DataLoader(
    dataset_train, shuffle=True, batch_size=32,
)