import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

print("Initializing AI Model Fine-Tuner Simulation...")

# 1. Create dummy data (e.g. for a regression or classification task)
X = torch.randn(1000, 10)  # 1000 samples, 10 features
y = torch.randint(0, 2, (1000,)) # Binary targets

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Define a simple Neural Network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Simulate fine-tuning loop
epochs = 5
print(f"Starting fine-tuning for {epochs} epochs...")

for epoch in range(epochs):
    running_loss = 0.0
    start_time = time.time()
    
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(dataloader):.4f} | Time: {epoch_time:.2f}s")

print("Fine-tuning completed successfully!")

# Save the model state
torch.save(model.state_dict(), 'fine_tuned_model.pth')
print("Model saved to fine_tuned_model.pth")
