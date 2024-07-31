import torch
import torch.nn as nn
from torchvision import models

class ResNetLSTMModel(nn.Module):
    def __init__(self, window_size, hidden_dim, num_keypoints):
        super(ResNetLSTMModel, self).__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_keypoints = num_keypoints
        
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.lstm = nn.LSTM(num_ftrs, hidden_dim, batch_first=True)
        self.fc_points = nn.Linear(hidden_dim, num_keypoints * 2)
        self.fc_presence = nn.Linear(hidden_dim, num_keypoints)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        x = x.view(batch_size * seq_len, 3, h, w)
        x = self.resnet(x)
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        
        points = self.fc_points(lstm_out)
        points = points.view(batch_size, self.num_keypoints, 2)
        
        presence = self.fc_presence(lstm_out)
        presence = self.sigmoid(presence)
        
        return points, presence
