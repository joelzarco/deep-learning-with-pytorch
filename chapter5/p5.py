#p5
# Saving a loading pre-trained models
# Save the pre-trained model as ModelCNN.pth remembering to save the weights, not only the architecture
# Create a model instance called loaded_model from the class ManufacturingCNN()
# Load ModelCNN.pth weights to loaded_model by passing the weights to .load_state_dict()


# Save the model
torch.save(model.state_dict(), 'ModelCNN.pth')

# Create a new model
loaded_model = ManufacturingCNN()

# Load the saved model
loaded_model.load_state_dict(torch.load('ModelCNN.pth'))
print(loaded_model)

'''
    ManufacturingCNN(
      (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU()
      (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (fc1): Linear(in_features=4096, out_features=4, bias=True)
    )
'''