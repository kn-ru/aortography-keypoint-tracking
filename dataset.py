import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class KeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, window_size=10):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.window_size = window_size
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.data = self._load_data()
        self.sequence_indices = self._create_sequence_indices()
        self.keypoint_order = [
            'AA1', 'AA2', 'STJ1', 'STJ2', 'CP', 'CM', 'CD', 'CT', 'PT', 'FE1', 'FE2'
        ]
    
    def _load_data(self):
        data = {}
        for image_file in self.image_files:
            identifier = image_file.split('_')[0]
            if identifier not in data:
                data[identifier] = {'images': [], 'annotations': []}
            image_path = os.path.join(self.image_dir, image_file)
            annotation_path = os.path.join(self.annotation_dir, image_file + '.json')
            
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            
            data[identifier]['images'].append(image_path)
            data[identifier]['annotations'].append(annotation['objects'])
        return data
    
    def _create_sequence_indices(self):
        sequence_indices = []
        for key in self.data.keys():
            num_images = len(self.data[key]['images'])
            for i in range(num_images - self.window_size):
                sequence_indices.append((key, i))
        return sequence_indices
    
    def __len__(self):
        return len(self.sequence_indices)
    
    def __getitem__(self, idx):
        key, start_idx = self.sequence_indices[idx]
        image_paths = self.data[key]['images'][start_idx:start_idx + self.window_size]
        next_image_path = self.data[key]['images'][start_idx + self.window_size]
        
        images = [Image.open(img_path).convert('L') for img_path in image_paths]
        next_image = Image.open(next_image_path).convert('L')
        
        if self.transform:
            images = [self.transform(img) for img in images]
            next_image = self.transform(next_image)
        
        images = torch.stack(images)
        
        points = []
        for ann in self.data[key]['annotations'][start_idx:start_idx + self.window_size]:
            frame_points = {kp: [0.0, 0.0] for kp in self.keypoint_order}
            for obj in ann:
                class_title = obj['classTitle']
                if class_title in frame_points:
                    x, y = obj['points']['exterior'][0]
                    frame_points[class_title] = [x, y]
            points.append([frame_points[kp] for kp in self.keypoint_order])
        
        next_points = {kp: [0.0, 0.0] for kp in self.keypoint_order}
        for obj in self.data[key]['annotations'][start_idx + self.window_size]:
            class_title = obj['classTitle']
            if class_title in next_points:
                x, y = obj['points']['exterior'][0]
                next_points[class_title] = [x, y]
        
        points = torch.tensor(points, dtype=torch.float)
        next_points = torch.tensor([next_points[kp] for kp in self.keypoint_order], dtype=torch.float)
        num_points = torch.tensor((next_points != torch.tensor([0.0, 0.0], device=next_points.device).clone().detach()).all(dim=1).sum().item(), dtype=torch.long)

        
        return images, points, next_image, next_points, num_points

def custom_collate_fn(batch):
    images, points, next_images, next_points, num_points = zip(*batch)
    
    images = torch.stack(images)
    next_images = torch.stack(next_images)
    
    points = torch.stack(points)
    next_points = torch.stack(next_points)
    num_points = torch.stack(num_points)
    
    return images, points, next_images, next_points, num_points
