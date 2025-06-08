import torch
import numpy as np
from model.trainer import LSTMModel

class GesturePredictor:
    def __init__(self, model_path="model.pth"):
        self.model = LSTMModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, sequence):
        with torch.no_grad():
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            output = self.model(x)
            return output.item() > 0.5    