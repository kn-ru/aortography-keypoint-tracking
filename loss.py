import torch
import torch.nn as nn

class CustomPoseLoss(nn.Module):
    def __init__(self, device, num_keypoints=11):
        super(CustomPoseLoss, self).__init__()
        self.device = device
        self.num_keypoints = num_keypoints
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, batch):
        images, points, next_image, next_points, num_points = batch
        pred_points, pred_presence = preds
        pred_points = pred_points.to(self.device)
        pred_presence = pred_presence.to(self.device)
        next_points = next_points.to(self.device)

        mask = (next_points != 0).all(dim=-1).float()

        keypoints_loss = self.mse_loss(pred_points[mask == 1], next_points[mask == 1])
        presence_loss = self.bce_loss(pred_presence, mask)
        
        total_loss = keypoints_loss + presence_loss

        return total_loss
