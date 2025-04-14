#p4
#Optimizers and training loop

import torch.optim as optim

net = Net()

# Define the Adam optimizer
optimizer = optim.Adagrad(net.parameters(), lr=0.001)

train_model(
    optimizer=optimizer,
    net=net,
    num_epochs=10,
)

### other optios are SGD and MSRprop
# Define the RMSprop optimizer
#optimizer = optim.RMSprop(net.parameters(), lr=0.001)
#optimizer = optim.SGD(net.parameters(), lr=0.001)
