import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from dataloader import load_data
from models import SimpleCNN, ImprovedCNN
from trainer import train_model, evaluate_model, calculate_metrics
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


def visualize_grad_cam(model, img, label, target_layer):
    cam_extractor = GradCAM(model, target_layer)
    model.eval()

    img = img.unsqueeze(0).to(device)
    out = model(img)
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    result = overlay_mask(to_pil_image(img.squeeze(0).cpu()), to_pil_image(activation_map[0], mode='F'), alpha=0.5)
    plt.imshow(result)
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.savefig(f'grad_cam_{target_layer}.png')


if __name__ == "__main__":
    num_classes = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_dir = 'data/imgs/'
    train_csv = 'data/train_data.csv'
    val_csv = 'data/val_data.csv'
    test_csv = 'data/test_data.csv'

    base_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])

    train_loader, val_loader, test_loader = load_data(train_csv, val_csv, test_csv, img_dir, batch_size=32,
                                                      base_transform=base_transform, augment_transform=None)

    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

    print("\n----- Simple CNN model -----\n")

    lr = 0.001
    num_epochs = 10

    model = SimpleCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_model(model, train_loader, criterion, optimizer, device, num_epochs, scheduler=None)
    evaluate_model(model, val_loader, device)
    accuracy, precision, recall, f1 = calculate_metrics(model, test_loader, device)


# --- ImprovedCNN ---
    print("\n----- Improved CNN model -----\n")
    lr = 0.001
    num_epochs = 10

    model = ImprovedCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_model(model, train_loader, criterion, optimizer, device, num_epochs, scheduler=None)
    evaluate_model(model, val_loader, device)
    accuracy, precision, recall, f1 = calculate_metrics(model, test_loader, device)

    print("\n----- Improved CNN model + Hyperparameter tuning -----\n")
    lr = 0.003
    num_epochs = 20

    model = ImprovedCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    train_model(model, train_loader, criterion, optimizer, device, num_epochs, scheduler)
    evaluate_model(model, val_loader, device)
    accuracy, precision, recall, f1 = calculate_metrics(model, test_loader, device)

    print("\n----- Improved CNN model + Data augmentation -----\n")
    base_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    augment_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(150, padding=4),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_loader, val_loader, test_loader = load_data(train_csv, val_csv, test_csv, img_dir, batch_size=32,
                                                      base_transform=base_transform,
                                                      augment_transform=augment_transform)

    lr = 0.003
    num_epochs = 20

    model = ImprovedCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    train_model(model, train_loader, criterion, optimizer, device, num_epochs, scheduler)
    evaluate_model(model, val_loader, device)
    accuracy, precision, recall, f1 = calculate_metrics(model, test_loader, device)

    image, label = test_loader.dataset[0]
    image_data = image.permute(1, 2, 0).numpy()
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    plt.imsave('original.png', image_data)
    visualize_grad_cam(model, image, label.item(), target_layer='conv1')
    visualize_grad_cam(model, image, label.item(), target_layer='conv2')
    visualize_grad_cam(model, image, label.item(), target_layer='conv3')
    visualize_grad_cam(model, image, label.item(), target_layer='conv4')

