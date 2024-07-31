import torch
import matplotlib.pyplot as plt
from dataset import KeypointDataset, custom_collate_fn
from models import ResNetLSTMModel

def visualize_inference(images, true_points, pred_points, pred_presence, idx=0):
    height, width = images.shape[-2:]

    colors = {
        'AA1': 'purple', 'AA2': 'orange', 'STJ1': 'pink', 'STJ2': 'brown',
        'CP': 'yellow', 'CM': 'green', 'CD': 'blue', 'CT': 'red', 
        'PT': 'cyan', 'FE1': 'black', 'FE2': 'magenta'
    }
    class_titles = list(colors.keys())

    fig, ax = plt.subplots(figsize=(8, 8))

    next_img = images[idx, -1].permute(1, 2, 0).cpu().numpy()
    ax.imshow(next_img, cmap='gray')

    for i, point in enumerate(true_points[idx, -1].cpu().numpy()):
        x, y = point
        if not (x == 0 and y == 0):
            x = (x / 1000) * width
            y = (y / 1000) * height
            class_title = class_titles[i]
            ax.scatter(x, y, c=colors[class_title], s=50, label=class_title if class_title not in [t.get_text() for t in ax.texts] else "")
            ax.text(x, y, class_title, color=colors[class_title], fontsize=12)

    for i, (point, presence) in enumerate(zip(pred_points[idx].cpu().numpy(), pred_presence[idx].cpu().numpy())):
        x, y = point
        if presence > 0.1 and not (x == 0 and y == 0):
            x = (x / 1000) * width
            y = (y / 1000) * height
            class_title = class_titles[i]
            ax.scatter(x, y, c=colors[class_title], s=50, edgecolors='black', linewidth=1.5, label=class_title if class_title not in [t.get_text() for t in ax.texts] else "")
            ax.text(x, y, class_title, color='black', fontsize=12, bbox=dict(facecolor=colors[class_title], alpha=0.5))

    ax.axis('off')
    ax.set_title('Next Frame')
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')

    plt.tight_layout()
    plt.show()

image_dir = '/media/Data/MEDICAL/Aortography keypoint tracking for TAVI/images'
annotation_dir = '/media/Data/MEDICAL/Aortography keypoint tracking for TAVI/annotations'
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
test_dataset = KeypointDataset(image_dir=image_dir, annotation_dir=annotation_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=16, collate_fn=custom_collate_fn)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = ResNetLSTMModel(window_size=10, hidden_dim=256, num_keypoints=11).to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

with torch.no_grad():
    for images, true_points, next_image, next_points, num_points in test_dataloader:
        images, true_points, next_image, next_points, num_points = (
            images.to(device), true_points.to(device), next_image.to(device), 
            next_points.to(device), num_points.to(device)
        )

        pred_points, pred_presence = model(images)

        visualize_inference(images, true_points, pred_points, pred_presence, idx=0)
        break
