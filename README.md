# 🚦 Traffic Sign Classification System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)

A deep learning project for traffic sign recognition using transfer learning with MobileNetV2. This system classifies 12 different traffic signs with >90% accuracy and includes an interactive Streamlit dashboard for real-time testing and evaluation.

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🔧 Usage](#-usage)
- [📊 Model Performance](#-model-performance)
- [🎮 Interactive Dashboard](#-interactive-dashboard)
- [📁 Project Structure](#-project-structure)
- [🔬 Technical Details](#-technical-details)
- [🛠️ Development](#️-development)
- [📈 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

## 🎯 Project Overview

This project implements a robust traffic sign classification system using state-of-the-art deep learning techniques. The system leverages MobileNetV2's transfer learning capabilities to achieve high accuracy on a custom dataset of 12 traffic sign classes.

### Key Objectives
- **High Accuracy**: Achieve >90% classification accuracy across all traffic sign classes
- **Real-time Performance**: Optimized for fast inference suitable for real-world applications
- **User-friendly Interface**: Interactive dashboard for easy testing and evaluation
- **Scalable Architecture**: Modular design allowing for easy extension to new classes

## ✨ Features

- 🧠 **Deep Learning Model**: MobileNetV2 with transfer learning
- 📊 **Comprehensive Evaluation**: Detailed metrics, confusion matrix, and performance analysis
- 🖥️ **Interactive Dashboard**: Three-mode Streamlit application for testing
- 📈 **Real-time Inference**: Fast prediction on single images
- 📁 **Organized Dataset**: Structured data pipeline with preprocessing
- 🎯 **Class-specific Testing**: Evaluate performance per traffic sign category
- 📉 **Visual Analytics**: Comprehensive graphs and performance visualizations

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Image   │───▶│  Preprocessing   │───▶│  MobileNetV2    │
│   (224×224×3)   │    │ (Resize + Norm)  │    │  (Frozen)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Prediction    │◀───│   Dense Layer    │◀───│ Global Avg Pool │
│   (12 Classes)  │    │   (Softmax)      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/traffic-sign-classification.git
cd traffic-sign-classification

# Install dependencies
pip install -r requirements.txt

# Train the model
cd src
python train.py

# Launch the dashboard
cd ../app
streamlit run app.py
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/traffic-sign-classification.git
   cd traffic-sign-classification
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import tensorflow; print('TensorFlow version:', tensorflow.__version__)"
   ```

### Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| TensorFlow | 2.13.0 | Deep learning framework |
| OpenCV | Latest | Image processing |
| NumPy | Latest | Numerical computations |
| Scikit-learn | Latest | Machine learning utilities |
| Matplotlib | Latest | Plotting and visualization |
| Streamlit | Latest | Web application framework |
| Seaborn | Latest | Statistical visualization |
| TQDM | Latest | Progress bars |

## 🔧 Usage

### Training the Model

```bash
cd src
python train.py
```

**Training Parameters:**
- Epochs: 8 with early stopping
- Batch size: 32
- Learning rate: 1e-4
- Optimizer: Adam
- Loss function: Sparse categorical crossentropy

### Model Evaluation

```bash
cd src
python evaluate.py
```

**Evaluation Metrics:**
- Overall accuracy
- Per-class accuracy
- Confusion matrix
- Classification report
- Performance visualizations

### Interactive Dashboard

```bash
cd app
streamlit run app.py
```

Access the dashboard at `http://localhost:8501`

## 📊 Model Performance

### Classification Classes

| Class | Description | Sample Count |
|-------|-------------|--------------|
| `children` | Children crossing warning | - |
| `no_entry` | No entry prohibition | - |
| `pedestrian` | Pedestrian crossing | - |
| `road_work` | Road work ahead | - |
| `speed_30` | Speed limit 30 km/h | - |
| `speed_50` | Speed limit 50 km/h | - |
| `speed_70` | Speed limit 70 km/h | - |
| `speed_80` | Speed limit 80 km/h | - |
| `stop` | Stop sign | - |
| `turn_left` | Turn left indication | - |
| `turn_right` | Turn right indication | - |
| `yield` | Yield to oncoming traffic | - |

### Performance Metrics

- **Overall Accuracy**: >90%
- **Training Time**: ~15 minutes (GPU) / ~2 hours (CPU)
- **Inference Time**: <50ms per image
- **Model Size**: ~14MB

## 🎮 Interactive Dashboard

The Streamlit dashboard offers three distinct testing modes:

### 🖼️ Mode 1: Single Image Testing
- Upload individual traffic sign images
- Real-time prediction with confidence scores
- Visual feedback on prediction reliability
- Preprocessing visualization

### 📊 Mode 2: Class-wise Testing
- Select specific traffic sign classes
- Batch testing on multiple images
- Success/failure categorization
- Per-class accuracy metrics

### 📈 Mode 3: Comprehensive Evaluation
- Full dataset evaluation
- Overall performance metrics
- Detailed class-wise analysis
- Performance trend visualization
- Problematic class identification

## 📁 Project Structure

```
traffic-sign-classification/
├── 📂 data/                          # Dataset directory
│   ├── 📂 raw/                       # Raw, unprocessed images
│   │   ├── 📂 GTSRB/                 # Original GTSRB dataset
│   │   └── 📂 selected_classes/      # Selected traffic sign classes
│   └── 📂 processed/                 # Preprocessed images (224×224)
│       ├── � children/              # Class-wise organized data
│       ├── 📂 no_entry/
│       ├── 📂 pedestrian/
│       ├── 📂 road_work/
│       ├── 📂 speed_30/
│       ├── 📂 speed_50/
│       ├── 📂 speed_70/
│       ├── 📂 speed_80/
│       ├── 📂 stop/
│       ├── 📂 turn_left/
│       ├── 📂 turn_right/
│       └── 📂 yield/
├── 📂 src/                           # Source code
│   ├── 🐍 data_loader.py             # Data loading and preprocessing
│   ├── 🧠 model.py                   # MobileNetV2 model architecture
│   ├── 🏋️ train.py                   # Model training script
│   ├── 📊 evaluate.py                # Model evaluation and metrics
│   ├── 📈 graph1.py                  # Performance visualization
│   └── 📉 graph2.py                  # Additional analytics
├── 📂 app/                           # Streamlit application
│   └── 🐍 app.py                     # Interactive dashboard
├── 📂 models/                        # Trained models
│   └── 💾 best_model.h5              # Saved TensorFlow model
├── 📂 results/                       # Evaluation outputs
│   ├── 📂 figures/                   # Performance graphs
│   └── 📂 metrics/                   # Detailed metrics
├── 📂 notebooks/                     # Jupyter notebooks (optional)
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # Project documentation
└── 📄 LICENSE                        # License file
```

## � Technical Details

### Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Shape**: 224×224×3 RGB images
- **Feature Extraction**: Frozen base layers
- **Classification Head**: Custom dense layers
- **Output Layer**: 12-class softmax activation

### Preprocessing Pipeline

1. **Image Resizing**: Scale to 224×224 pixels
2. **Color Conversion**: BGR to RGB format
3. **Normalization**: MobileNetV2-specific preprocessing
4. **Batch Generation**: Efficient data loading

### Training Strategy

- **Transfer Learning**: Leverage ImageNet pre-trained weights
- **Fine-tuning**: Freeze base layers, train classification head
- **Regularization**: Dropout (0.5) to prevent overfitting
- **Optimization**: Adam optimizer with reduced learning rate

## 🛠️ Development

### Code Style

This project follows PEP 8 Python style guidelines with additional conventions:

- **Docstrings**: Google-style documentation
- **Type Hints**: Optional for better code clarity
- **Variable Naming**: Descriptive, snake_case
- **Function Organization**: Logical grouping by functionality

### Testing

```bash
# Run unit tests (if available)
python -m pytest tests/

# Validate model loading
python -c "from src.model import build_model; print('Model builds successfully')"
```

### Contributing Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Results

### Training Performance

The model achieves consistent performance across multiple training runs:

- **Training Accuracy**: 95%+
- **Validation Accuracy**: 90%+
- **Loss Convergence**: Stable within 8 epochs
- **Generalization**: Good performance on unseen data

### Confusion Matrix Analysis

The confusion matrix reveals:
- **High Confidence Classes**: Stop, Yield, Speed limits
- **Challenging Distinctions**: Similar speed limit signs
- **Consistent Performance**: No significant class bias

### Performance Visualizations

- **Training Curves**: Loss and accuracy progression
- **Class-wise Accuracy**: Bar chart of per-class performance
- **Confusion Matrix**: Heatmap of prediction patterns
- **Error Analysis**: Misclassification examples

## 🤝 Contributing

We welcome contributions to improve this traffic sign classification system! Here's how you can help:

### Areas for Contribution

- 🆕 **New Classes**: Add support for additional traffic signs
- 🚀 **Performance**: Optimize inference speed and accuracy
- 🎨 **UI/UX**: Enhance the Streamlit dashboard
- 📚 **Documentation**: Improve guides and examples
- 🧪 **Testing**: Add comprehensive test coverage
- 🔧 **Deployment**: Containerize and deploy the application

### Development Workflow

1. **Setup Development Environment**
   ```bash
   git clone https://github.com/yourusername/traffic-sign-classification.git
   cd traffic-sign-classification
   python -m venv dev-env
   source dev-env/bin/activate  # or dev-env\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # if available
   ```

2. **Make Changes**
   - Follow existing code style
   - Add tests for new features
   - Update documentation

3. **Submit Changes**
   - Create descriptive commit messages
   - Ensure all tests pass
   - Submit pull request with clear description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### License Summary

- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❗ Liability disclaimer
- ❗ Warranty disclaimer

## 🙏 Acknowledgments

This project was made possible by:

- **GTSRB Dataset**: German Traffic Sign Recognition Benchmark
- **TensorFlow Team**: For the excellent deep learning framework
- **MobileNetV2 Authors**: For the efficient architecture
- **Streamlit Community**: For the intuitive web framework
- **OpenCV Contributors**: For powerful image processing tools

### References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_news.html)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📞 Contact

For questions, suggestions, or collaborations:

- **Project Maintainer**: Abderrahmane Foutouh 
- **Email**: [foutouhabderrahman8@gmail.com]
- **GitHub**: [https://github.com/abdoufoutouh]
- **LinkedIn**: [https://www.linkedin.com/in/foutouh-abderrahmane-537447305/]

---

⭐ If this project helps you, consider giving it a star! 🌟
