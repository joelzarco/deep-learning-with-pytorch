# p1
# two input dataset
# Build a custom dataset for the Omniglot problem that serves triplets consisting of:
# 	- The image of a character to be classified
#	- The OHE vector of length 30 with the ID of the character
#	- the target label, an integer between 0 and 963

class OmniglotDataset(Dataset):
    def __init__(self, transform, samples):
		# Assign transform and samples to class attributes
        self.transform = transform
        self.samples = samples
                    
    def __len__(self):
		# Return number of samples
        return len(self.samples)

    def __getitem__(self, idx):
      	# Unpack the sample at index idx
        img_path, alphabet, label = self.samples[idx]
        img = Image.open(img_path).convert('L')
        # Transform the image 
        img_transformed = self.transform(img)
        return img_transformed, alphabet, label