import torch
import torch.nn as nn
import os

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Save a dummy model if it doesn't exist
model_path = "models/mlp_model.pth"
if not os.path.exists(model_path):
    model = MLP()
    torch.save(model.state_dict(), model_path)
    print(f"✅ Dummy model saved at {model_path}")