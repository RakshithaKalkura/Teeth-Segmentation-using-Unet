import torch
from model import Model

# Instantiate the model
model = Model()

# Load the model weights
model_path = 'model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

print("Model loaded successfully!")