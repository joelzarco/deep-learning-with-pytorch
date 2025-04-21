#p4
# RNN training loop
# setup MSELoss as criterion, re shape seqs, get model outputs and calc loss

net = Net()
# Set up MSE loss
criterion = nn.MSELoss()
optimizer = optim.Adam(
  net.parameters(), lr=0.0001
)

for epoch in range(3):
    for seqs, labels in dataloader_train:
        # Reshape model inputs
        seqs = seqs.view(32, 96, 1)
        # Get model outputs
        outputs = net(seqs)
        # Compute loss
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

''' Loss is decreasing over the three epochs:

	Epoch 1, Loss: 0.703811526298523
    Epoch 2, Loss: 0.6919015645980835
    Epoch 3, Loss: 0.675562858581543
'''