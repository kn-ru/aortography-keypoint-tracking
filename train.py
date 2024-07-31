import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from dataset import KeypointDataset, custom_collate_fn
from models import ResNetLSTMModel
from loss import CustomPoseLoss

image_dir = '/media/Data/MEDICAL/Aortography keypoint tracking for TAVI/images'
annotation_dir = '/media/Data/MEDICAL/Aortography keypoint tracking for TAVI/annotations'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = KeypointDataset(image_dir=image_dir, annotation_dir=annotation_dir, transform=transform)

def get_series_identifiers(image_files):
    return sorted(set(file.split('_')[0] for file in image_files))

series_identifiers = get_series_identifiers(dataset.image_files)
train_series, test_series = train_test_split(series_identifiers, test_size=0.2, random_state=42)

def filter_dataset_by_series(dataset, series_list):
    indices = [i for i, (key, _) in enumerate(dataset.sequence_indices) if key in series_list]
    return torch.utils.data.Subset(dataset, indices)

train_dataset = filter_dataset_by_series(dataset, train_series)
test_dataset = filter_dataset_by_series(dataset, test_series)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=16, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=16, collate_fn=custom_collate_fn)

num_epochs = 200
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = ResNetLSTMModel(window_size=10, hidden_dim=256, num_keypoints=11).to(device)
loss_fn = CustomPoseLoss(device=device, num_keypoints=11)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=num_epochs)

losses = {'train_loss': [], 'val_loss': []}

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, points, next_image, next_points, num_points in train_dataloader:
        images, points, next_image, next_points, num_points = (
            images.to(device), points.to(device), next_image.to(device), 
            next_points.to(device), num_points.to(device)
        )
        
        optimizer.zero_grad()
        preds = model(images)
        loss = loss_fn(preds, (images, points, next_image, next_points, num_points))
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
    
    train_loss /= len(train_dataloader)
    losses['train_loss'].append(train_loss)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, points, next_image, next_points, num_points in test_dataloader:
            images, points, next_image, next_points, num_points = (
                images.to(device), points.to(device), next_image.to(device), 
                next_points.to(device), num_points.to(device)
            )
            
            preds = model(images)
            loss = loss_fn(preds, (images, points, next_image, next_points, num_points))
            val_loss += loss.item()
    
    val_loss /= len(test_dataloader)
    losses['val_loss'].append(val_loss)
    
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

print('Finished Training')
