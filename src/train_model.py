import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from snn_model import WeldingSNN

# Configuration
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 50
# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"Training on device: {DEVICE}")
    
    # 1. Load Data
    # In a real run, you would load the .npy files saved by preprocessing
    print("Loading dataset...")
    # Creating dummy data for demonstration (100 samples, 13 features)
    X_train = torch.randn(100, 13).to(DEVICE) 
    # Add time dimension for SNN: [Batch, Time_Steps, Features]
    # We repeat the features across 5 time steps
    X_train = X_train.unsqueeze(1).repeat(1, 5, 1)
    
    y_train = torch.randint(0, 5, (100,)).to(DEVICE) # 5 Classes

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize Model
    model = WeldingSNN(num_inputs=13, num_classes=5).to(DEVICE)

    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print("Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in loader:
            optimizer.zero_grad()
            
            # Forward pass (SNN processes time steps internally)
            outputs = model(inputs) 
            
            # SpikingJelly usually returns [Time_Steps, Batch, Classes]
            # We average over time steps to get final prediction
            if outputs.dim() == 3: 
                outputs = outputs.mean(0) 
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

    # 5. Save the Model
    torch.save(model.state_dict(), "welding_snn_model.pth")
    print("Model saved as welding_snn_model.pth")

if __name__ == "__main__":
    train()
