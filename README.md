# Medical Image Enhancement and Segmentation

## Problem Statement
This project focuses on developing models for medical image enhancement and segmentation to improve diagnostic accuracy and clinical outcomes.

## Three Model Approaches - Performance Comparison

### 1️⃣ U-Net (CNN-based)
**Type**: Convolutional Neural Network  
**Characteristics**:
- ✅ Fast inference time
- ✅ Low memory footprint
- ✅ Proven track record for medical imaging
- ❌ Limited long-range dependencies
- ❌ May struggle with complex patterns

**Best For**: Production deployment, resource-constrained environments

### 2️⃣ Vision Transformer (ViT)
**Type**: Transformer-based  
**Characteristics**:
- ✅ Captures long-range dependencies
- ✅ State-of-the-art accuracy
- ✅ Better for complex anatomical structures
- ❌ Higher computational cost
- ❌ Requires more training data

**Best For**: High-accuracy requirements, research applications

### 3️⃣ U-Net with Attention (Hybrid)
**Type**: CNN + Attention Mechanisms  
**Characteristics**:
- ✅ Balanced performance and efficiency
- ✅ Better feature selection than pure CNN
- ✅ Lower computational cost than ViT
- ⚠️ Moderate complexity

**Best For**: Production systems with moderate accuracy requirements

## Setup Instructions

### 1. Clone and Setup
```bash
git clone https://github.com/madhusoodansm-ux/hackton-.git
cd hackton-
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Run Model Comparison

```bash
python scripts/compare_models.py
```

### Output
The script will generate:
- 📊 Comparison metrics (size, inference time, memory)
- 📈 Visualization charts
- 📋 JSON report with detailed results
- 💡 Recommendations for each use case

## Metrics Evaluated

### Performance Metrics
- **Dice Coefficient**: Overlap between predicted and ground truth
- **IoU (Intersection over Union)**: Segmentation accuracy
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **Hausdorff Distance**: Boundary accuracy

### Efficiency Metrics
- **Model Size**: Disk space and deployment size
- **Inference Time**: Speed of predictions (ms per image)
- **Peak Memory**: Maximum GPU/CPU memory usage
- **Parameter Count**: Total trainable parameters

## Project Structure
```
├── config.py                    # Configuration for all models
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── models/
│   ├── __init__.py
│   ├── unet.py                 # U-Net implementation
│   ├── vit.py                  # Vision Transformer
│   └── unet_attention.py        # U-Net + Attention
├── utils/
│   ├── __init__.py
│   └── metrics.py              # Segmentation metrics
├── scripts/
│   └── compare_models.py        # Comparison script
├── data/                        # Dataset directory
├── models_saved/               # Trained model checkpoints
├── results/                    # Comparison results & visualizations
└── logs/                       # Training logs
```

## Dataset

- **BRATS Dataset**: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
- **Google Drive**: https://drive.google.com/drive/folders/19psnKwO4swOQ6BUE2xuPpxu0PyJ3SKM1
- **Image Types**: T1, T1ce, T2, FLAIR (4 modalities)
- **Segmentation Classes**: Background, NCR/NET, Edema, Enhancing Tumor

## Next Steps

1. **Download Dataset**
   - Download BRATS dataset and place in `data/` directory

2. **Prepare Data**
   - Run preprocessing pipeline
   - Create train/val/test splits

3. **Train Models**
   - Train all three models with same hyperparameters
   - Monitor metrics during training

4. **Evaluate**
   - Run comparison script
   - Analyze results

5. **Deploy**
   - Select best model for your use case
   - Export for deployment

## Configuration

Edit `config.py` to customize:
- Model architectures
- Training hyperparameters
- Data preprocessing
- Augmentation strategies
- Evaluation metrics

## Results Interpretation

### When to Use Each Model

**Choose U-Net if**:
- You need fast inference (real-time)
- Deployment on edge devices
- Limited GPU/CPU resources
- Clear, well-defined anatomical boundaries

**Choose Vision Transformer if**:
- You have large dataset and computation resources
- Need maximum accuracy
- Complex anatomical structures
- Can afford longer inference times

**Choose U-Net + Attention if**:
- You want balance between speed and accuracy
- Production environment with moderate requirements
- You need attention visualization
- Resource constraints are moderate

## Usage in VS Code

1. Open VS Code
2. Open integrated terminal: `Ctrl + ~`
3. Activate virtual environment
4. Run comparison: `python scripts/compare_models.py`
5. View results in `results/` directory
6. Visualizations auto-open in matplotlib

## Contributing

Feel free to contribute improvements:
- New architectures
- Better preprocessing
- Additional metrics
- Optimization techniques

## License

This project is open source and available under the MIT License.

## Contact

- **Author**: Madhusoodan SM
- **Email**: madhusoodansm@jnnce.ac.in
- **GitHub**: https://github.com/madhusoodansm-ux
