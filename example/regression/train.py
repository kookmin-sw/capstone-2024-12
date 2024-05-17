import torch
import torch.nn as nn
import torch.optim as optim

from model import ModelClass

x = torch.randn(100, 1) * 10
y = x + 3 * torch.randn(100,1)
    
model = ModelClass()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

torch.save(model, './model.pt')