# 🎯 ResNet-50 CIFAR-100 Image Classifier

<div align="center">

**State-of-the-Art Deep Learning Image Classifier Achieving 81.45% Top-1 Accuracy**

[Live Demo on Huggingface Spaces](https://huggingface.co/spaces/agentic-snehanshu/resnet50-cifar100)

</div>

***

## 🌟 Highlights

- 🏆 **81.45% Top-1 Accuracy** - Exceeds 73% target by **8.45%**
- 🚀 **95.70% Top-5 Accuracy** - Model correctly identifies class in top-5 predictions 95.7% of the time
- ⚡ **Trained on NVIDIA A100 80GB** - Leveraging cutting-edge hardware for optimal performance
- 🎨 **100 Fine-Grained Classes** - Covering animals, vehicles, household items, nature scenes, and more
- 🌐 **Live Interactive Demo** - Try it instantly on HuggingFace Spaces
- 📦 **Production-Ready** - Deployed Gradio web application with clean inference pipeline

***

## 📊 Results

### Performance Metrics

| Metric | Score |
| :-- | :-- | :-- |
| **Top-1 Accuracy** | **81.45%**  |
| **Top-5 Accuracy** | **95.70%**  |
| Target Accuracy | 73.00% |
| **Improvement** | **+8.45%** |
| Training Time | ~1 hours |
| Test Images Classified | 10,000 |
| Correctly Classified (Top-1) | 8,145 |

### Training Progression

```
Epoch   1:   1.06% accuracy  (Learning rate warmup begins)
Epoch  10:  ~20% accuracy    (Rapid initial learning)
Epoch  50:  ~58% accuracy    (Steady improvement)
Epoch 100:  ~68% accuracy    (Approaching plateau)
Epoch 150:  ~75% accuracy    (Fine-tuning phase)
Epoch 200:  81.45% accuracy  (Final convergence) ✅
```

***

## 🏗️ Model Architecture

### ResNet-50 (Adapted for CIFAR-100)

<details>
```
<summary><b>Architecture Details</b> (Click to expand)</summary>
```

```
Input: 32×32×3 RGB Image
    ↓
Conv1: 3×3 conv, 64 filters, stride=1 (CIFAR adaptation)
    ↓
Layer 1: 3 × Bottleneck (64 → 256 channels)
    ↓
Layer 2: 4 × Bottleneck (256 → 512 channels, stride=2)
    ↓
Layer 3: 6 × Bottleneck (512 → 1024 channels, stride=2)
    ↓
Layer 4: 3 × Bottleneck (1024 → 2048 channels, stride=2)
    ↓
Global Average Pooling (2048 features)
    ↓
Fully Connected: 2048 → 100 classes
    ↓
Output: 100-dimensional probability distribution
```

**Key Modifications for CIFAR-100:**
- Initial 3×3 convolution (vs. 7×7 for ImageNet)
- No max pooling after first conv (preserves spatial resolution)
- Stride=1 in initial conv (maintains 32×32 feature maps)

**Model Statistics:**
- **Total Parameters**: 23,528,100
- **Trainable Parameters**: 23,528,100
- **Model Size**: 92 MB (FP32)
- **Depth**: 50 layers
- **Bottleneck Blocks**: 16 (3+4+6+3)

</details>

***

## 🔬 Training Methodology

### Hardware \& Environment

- **GPU**: NVIDIA A100 80GB PCIe
- **VRAM Utilization**: ~40 GB (50% of available)
- **Batch Size**: 2048 (large batch training)
- **Training Duration**: ~1 hour (200 epochs)


### Hyperparameters

<details>
<summary><b>Detailed Configuration</b></summary>

#### Optimization
- **Optimizer**: SGD with Nesterov Momentum
  - Momentum: 0.9
  - Weight Decay: 5e-4
  - Nesterov: True
  
- **Learning Rate Schedule**: Cosine Annealing with Warmup
  - Initial LR: 0.8 (scaled for batch size 2048)
  - Base LR (batch 256): 0.1
  - Scaling Factor: 8× (0.1 × 2048/256)
  - Warmup Epochs: 10
  - Min LR: 1e-6
  - Schedule: Cosine decay after warmup

#### Training Configuration
- **Epochs**: 200
- **Batch Size**: 2048
- **Gradient Accumulation**: None (large single batch)
- **Mixed Precision**: FP16 via GradScaler
- **Loss Function**: Cross-Entropy with Label Smoothing (ε=0.1)

</details>

### Data Augmentation Pipeline

Our aggressive augmentation strategy prevents overfitting and improves generalization:

#### 1. **Geometric Transformations**

```python
- RandomResizedCrop(32×32, scale=(0.8, 1.0), ratio=(0.9, 1.1))
- RandomHorizontalFlip(p=0.5)
```


#### 2. **Cutout Regularization**

```python
- Cutout(n_holes=1, length=16)
- Randomly masks 16×16 patches
- Applied to 50% of images
```


#### 3. **AutoAugment**

```python
- AutoAugment Policy: CIFAR-10
- Applies learned augmentation strategies
- Operations: ShearX, TranslateY, Rotate, Color, Posterize, etc.
- Applied to 70% of images
```


#### 4. **Mixup Data Augmentation**

```python
- Mixup Alpha: 0.2
- Blends two images with interpolation
- Creates convex combinations: λ × img1 + (1-λ) × img2
- Regularization effect
```


#### 5. **Normalization**

```python
- Mean: [0.5071, 0.4867, 0.4408] (CIFAR-100 statistics)
- Std:  [0.2675, 0.2565, 0.2761]
```


### Regularization Techniques

1. **Label Smoothing** (ε=0.1)
    - Prevents overconfident predictions
    - Softens one-hot labels: (1-ε) for correct class, ε/(C-1) for others
2. **Weight Decay** (5e-4)
    - L2 regularization on network weights
    - Prevents overfitting
3. **Mixup** (α=0.2)
    - Trains on linear interpolations of examples
    - Improves calibration and robustness
4. **Data Augmentation**
    - Multiple random transformations
    - Increases effective dataset size
