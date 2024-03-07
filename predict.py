from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet

def predict(demo_image):
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load('checkpoints/best_model.pth')
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    # 设置数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.2, 0.2]),
    ])
    dataset = datasets.ImageFolder(root='data/classifiy/augtrain', transform=transform)
    classes = dataset.classes
    
    demo_input = transform(demo_image).unsqueeze(0).to(device)
    output = model(demo_input)
    _, predicted = torch.max(output, 1)
    id_cls = predicted.item()
    return classes[id_cls]

if __name__ == '__main__':
    demo_image_path = 'data/classifiy/augtrain/0/2_1.bmp'
    demo_image =  Image.open(demo_image_path).convert('RGB')
    print(predict(demo_image))