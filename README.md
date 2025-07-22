# Emotion Analysis with FLAVA

This repository contains a Jupyter Notebook implementation of a multimodal emotion analysis model using Facebook's FLAVA (FLexible and General-purpose Multimodal Transformer) model. The project analyzes emotions (positive, neutral, negative) from images and text, mimicking real-world applications like social media content analysis.

## Overview

- **Model**: FLAVA (facebook/flava-full) - a multimodal Transformer that processes both images and text.
- **Task**: Predict emotion labels (2 = Positive, 1 = Neutral, 0 = Negative) from multimodal inputs.
- **Dataset**: Custom dataset stored in Google Drive (`/content/drive/MyDrive/emotion_analysis/`), including `labels.csv` and corresponding images.
- **Evaluation Metrics**: Accuracy, F1 Score, and Confusion Matrix visualization.

## Files

- **`emotion-analysis.ipynb`**: The main Jupyter Notebook containing the complete code for data loading, model training, evaluation, and visualization.
  - Includes data preprocessing, FLAVA model setup, training loop (10 epochs), and performance metrics.
  - Saves the best model weights to `best_model.pth` (not uploaded due to size limits).

## Requirements

To run this project, you need the following dependencies:
- **Python 3.x**
- **PyTorch** (`torch>=2.7.1`)
- **Transformers** (`transformers>=4.53.2`)
- **Torchvision** (`torchvision>=0.22.1`)
- **scikit-learn** (`scikit-learn`)
- **seaborn** (`seaborn`)
- **matplotlib** (`matplotlib`)
- **pandas** (`pandas`)
- **Pillow** (`Pillow`)

Install the dependencies using:
```bash
!pip install --upgrade torch torchvision transformers scikit-learn seaborn matplotlib pandas pillow
```

# Setup

1. Google Colab Environment:
- Run the notebook in Google Colab for GPU support (e.g., T4, L4, or A100).
- Mount your Google Drive to access the dataset:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Dataset Preparation:
- Place your dataset in Google Drive under /content/drive/MyDrive/emotion_analysis/.
- Ensure labels.csv contains columns image_name, text_corrected, and overall_sentiment with sentiment labels mapped as:
  - very_positive or positive → 2
  - neutral → 1
  - very_negative or negative → 0
- Images should be in /content/drive/MyDrive/emotion_analysis/images/.

3. Hardware:
- Recommended: GPU (e.g., T4 or L4) for faster training. Check availability in Colab settings.

# Usage

1.Run the Notebook:
- Open emotion-analysis.ipynb in Google Colab.
- Execute all cells sequentially to:
  - Install dependencies.
  - Load and preprocess the dataset.
  - Train the FLAVA model for 10 epochs.
  - Evaluate on validation and test sets, generating accuracy, F1 score, and confusion matrix visualizations.
  - Save the best model weights.
2.Output:
-Metrics: Printed accuracy, F1 score, and confusion matrix for validation and test sets.
-Visualization: Plots showing validation accuracy and F1 score trends over epochs.
-Model: Best model saved as best_model.pth in /content/drive/MyDrive/emotion_analysis/.

# Results
- Epoch 1 Example (from notebook output):
  - Validation Accuracy: 0.6094
  - Validation F1 Score: 0.4616
  - Confusion Matrix: All samples predicted as Positive, indicating potential data imbalance or early training stage.
- Full results available after completing 10 epochs.

# Notes
- Training Time: Initial epoch may take ~2.5 hours on T4 GPU due to dataset size and FLAVA complexity. Optimize with smaller batch_size (e.g., 4) or num_workers (e.g., 2).
- Data Issues: Check for image loading failures (None samples) reported in logs.
- Improvements: Consider weighted loss for class imbalance or hyperparameter tuning (e.g., learning rate).
License

# Acknowledgments
- Based on Facebook's FLAVA model from Hugging Face.
- Utilizes Google Colab for computation.

# 中文版 (Simplified Chinese Version)

# 项目简介
本仓库包含一个使用 Facebook FLAVA 模型的多模态情感分析 Jupyter Notebook 实现。项目通过图像和文本预测情感（积极=2、中性=1、消极=0），模拟社交媒体内容分析等应用。

# 文件
emotion-analysis.ipynb: 主笔记本，包含数据加载、模型训练、评估和可视化代码。

# 依赖
- Python 3.x
- PyTorch, Transformers, Torchvision, scikit-learn, seaborn, matplotlib, pandas, Pillow
- 安装命令：!pip install --upgrade torch torchvision transformers scikit-learn seaborn matplotlib pandas pillow

#使用步骤
1. 在 Google Colab 中打开笔记本。
2. 挂载 Google Drive：from google.colab import drive; drive.mount('/content/drive')。
3. 将数据集放入 /content/drive/MyDrive/emotion_analysis/，运行所有单元格完成训练和评估。

# 注意事项
- 训练可能耗时长（T4 GPU 下约2.5小时/epoch），可优化 batch_size 和 num_workers。
- 检查图像加载失败日志。
