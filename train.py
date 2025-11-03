# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import MedCareCNN
from data_loader import get_data_loaders

# ✅ Step 1: Configuration
data_dir = "data"   # Path to your data folder
batch_size = 8
epochs = 5
learning_rate = 0.001

# ✅ Step 2: Load Data
train_loader, test_loader = get_data_loaders(data_dir, batch_size)

# ✅ Step 3: Initialize Model
model = MedCareCNN()

# ✅ Step 4: Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ✅ Step 5: Training Loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# ✅ Step 6: Save the trained model
torch.save(model.state_dict(), "medcare_model.pth")
print("✅ Model training completed and saved as 'medcare_model.pth'")
