import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# CIFAR-100 class names
CLASS_NAMES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# Model Architecture
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet50_CIFAR(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet50_CIFAR, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride))
            self.in_planes = planes * Bottleneck.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50_CIFAR(num_classes=100)
model.load_state_dict(torch.load('resnet50_cifar100_hf.pth', map_location=device, weights_only=False))
model.to(device)
model.eval()
print(f"Model loaded on {device}")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

def predict(image):
    if image is None:
        return None
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    results = {CLASS_NAMES[idx]: float(prob) for idx, prob in zip(top5_idx, top5_prob)}
    
    return results

examples = []
if os.path.exists('examples'):
    example_files = [f for f in os.listdir('examples') if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    examples = [[os.path.join('examples', f)] for f in sorted(example_files)]
    print(f"Found {len(examples)} example images")

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
    title="🎯 ResNet-50 CIFAR-100 Classifier (81.45% Accuracy)",
    description="""
    ### Upload any image for classification!
    
    **Model achieves 81.45% Top-1 accuracy on CIFAR-100 test set**
    
    **Architecture:** ResNet-50 adapted for CIFAR-100 (23M parameters)  
    **Training:** 200 epochs on NVIDIA A100 80GB  
    **Techniques:** Mixup, AutoAugment, Cutout, Label Smoothing
    
    Try the examples below or upload your own image!
    """,
    article="""
    ### Training Details
    - **Optimizer**: SGD with Nesterov momentum (0.9)
    - **Learning Rate**: Cosine annealing with warmup
    - **Batch Size**: 2048 (scaled LR: 0.8)
    - **Augmentations**: RandomResizedCrop, Mixup, Cutout, AutoAugment
    - **Regularization**: Label smoothing (0.1), Weight decay (5e-4)
    - **Training Time**: ~2 hours
    
    ### Performance
    - **Top-1 Accuracy**: 81.45%
    - **Top-5 Accuracy**: 95.70%
    - **Target Exceeded**: +8.45%
    """,
    examples=examples if examples else None,
    theme=gr.themes.Soft(),
    allow_flagging="never",
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()
