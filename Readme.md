---
title: ResNet-50 CIFAR-100 Classifier
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🎯 ResNet-50 CIFAR-100 Image Classifier

**81.45% Top-1 Accuracy | 95.70% Top-5 Accuracy**

## 🚀 Model Description

ResNet-50 trained from scratch on CIFAR-100, exceeding the 73% accuracy target by **8.45%**!

### Architecture
- **Model**: ResNet-50 (adapted for CIFAR-100)
- **Parameters**: 23 million
- **Input**: 32×32 RGB images (auto-resized)
- **Output**: 100 classes

### Training Techniques
- **Data Augmentation**: RandomResizedCrop, Mixup (α=0.2), Cutout (16×16), AutoAugment
- **Optimization**: SGD + Nesterov momentum, Cosine annealing with 10-epoch warmup
- **Regularization**: Label smoothing (0.1), Weight decay (5e-4)
- **Batch Size**: 2048 (scaled learning rate: 0.8)

### Training Setup
- **Hardware**: NVIDIA A100 80GB
- **Epochs**: 200
- **Training Time**: ~2 hours
- **Dataset**: CIFAR-100 (50k train, 10k test)

## 📊 Performance

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | **81.45%** |
| Top-5 Accuracy | **95.70%** |
| Target Accuracy | 73.00% |
| **Improvement** | **+8.45%** |

## 🎨 CIFAR-100 Categories

100 fine-grained classes across 20 superclasses:

- **Animals**: bear, leopard, lion, tiger, wolf, beaver, otter, seal, whale, etc.
- **Vehicles**: bicycle, bus, motorcycle, pickup_truck, train, streetcar, tractor, etc.
- **Household**: bed, chair, couch, table, wardrobe, clock, lamp, telephone, etc.
- **Nature**: cloud, forest, mountain, plain, sea, bridge, road, etc.
- **Plants**: flowers (orchid, poppy, rose, sunflower, tulip), trees (maple, oak, palm, pine, willow)
- **Food**: apple, orange, pear, sweet_pepper, mushroom
- **And more!**

## 🎯 Usage

Upload any image and get instant Top-5 predictions with confidence scores!
