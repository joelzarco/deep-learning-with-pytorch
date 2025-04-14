#p2
# Using dataset class defined in previous lesson create a DataLoader from path 'water_train.csv'
# DataLoader has been imported

# Create an instance of the WaterDataset
dataset_train = WaterDataset('water_train.csv')

# Create a DataLoader based on dataset_train
dataloader_train = DataLoader(
    dataset_train,
    batch_size=2,
    shuffle=True,
)

# Get a batch of features and labels
features, labels = next(iter(dataloader_train))
print(features, labels)