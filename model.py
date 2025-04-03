import torch
import torch.nn as nn
import torchvision.models as models
from dataset import group_1, group_2, group_3

class GroupKeypointModel(nn.Module):
    def __init__(self):
        super(GroupKeypointModel, self).__init__()
        # Используем предобученную ResNet34 в качестве бэкбона (более мощная модель)
        self.backbone = models.resnet34(pretrained=True)
        num_features = self.backbone.fc.in_features
        # Заменяем последний слой на Identity, чтобы получить вектор признаков
        self.backbone.fc = nn.Identity()
        
        # Общие признаки для всех групп
        self.shared_features = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.LayerNorm(512),  # Заменяем BatchNorm1d на LayerNorm, который работает с любым размером батча
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Отдельная голова для классификации групп
        self.group_classifier = nn.Linear(512, 3)  # 3 группы точек
        
        # Отдельные головы для каждой группы точек
        self.group1_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * len(group_1))  # (p, x, y) для каждой точки в группе 1
        )
        
        self.group2_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * len(group_2))  # (p, x, y) для каждой точки в группе 2
        )
        
        self.group3_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * len(group_3))  # (p, x, y) для каждой точки в группе 3
        )
        
        # Сохраняем размеры групп
        self.group1_size = len(group_1)
        self.group2_size = len(group_2)
        self.group3_size = len(group_3)
        self.num_keypoints = self.group1_size + self.group2_size + self.group3_size

    def forward(self, x):
        # Извлекаем признаки из бэкбона
        backbone_features = self.backbone(x)
        
        # Получаем общие признаки
        shared = self.shared_features(backbone_features)
        
        # Предсказываем вероятности групп
        group_probs = self.group_classifier(shared)
        
        # Предсказываем точки для каждой группы
        group1_out = self.group1_head(shared).view(-1, self.group1_size, 3)
        group2_out = self.group2_head(shared).view(-1, self.group2_size, 3)
        group3_out = self.group3_head(shared).view(-1, self.group3_size, 3)
        
        # Объединяем выходы всех групп
        # Формат выхода: [batch, num_keypoints, 3]
        # где 3 - это [presence_logit, x, y]
        all_keypoints = torch.cat([group1_out, group2_out, group3_out], dim=1)
        
        return {
            'keypoints': all_keypoints,  # [batch, num_keypoints, 3]
            'group_probs': group_probs,  # [batch, 3]
            'group1': group1_out,        # [batch, group1_size, 3]
            'group2': group2_out,        # [batch, group2_size, 3]
            'group3': group3_out         # [batch, group3_size, 3]
        }


# Для обратной совместимости с существующим кодом
class MultiHeadKeypointModel(GroupKeypointModel):
    def __init__(self, num_keypoints):
        super(MultiHeadKeypointModel, self).__init__()
        
    def forward(self, x):
        output = super(MultiHeadKeypointModel, self).forward(x)
        return output['keypoints']  # Возвращаем только предсказания точек для совместимости
