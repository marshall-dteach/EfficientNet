import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from efficientnet_pytorch import EfficientNet

device ='cuda' if torch.cuda.is_available() else 'cpu'
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
model.to(device)
# 设置数据变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.2, 0.2]),
])

# 加载数据集
dataset = datasets.ImageFolder(root='data/classifiy/augtrain', transform=transform)
# 打印类别名称及其对应的索引
print("Class indices:")
print(dataset.class_to_idx)
# 打印所有类别的名称
print("\nClass names:")
print(dataset.classes)
# 划分数据集为训练集和测试集
train_size = int(0.8 * len(dataset))  # 例如，使用80%的数据作为训练集
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 10
for epoch in range(epochs):
    all_preds = []
    all_targets = []
    model.train()
    for batch, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch}/{len(train_loader)}], Loss: {loss.item():.4f}')

    model.eval()
    best_acc = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            _, predicted = torch.max(preds.data, 1)
            # 收集所有预测和真实标签
            all_preds.extend(predicted.view(-1).cpu().numpy())
            all_targets.extend(labels.view(-1).cpu().numpy())
            
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_targets, all_preds)
    print(f'Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), 'checkpoints/best_model.pth')
    