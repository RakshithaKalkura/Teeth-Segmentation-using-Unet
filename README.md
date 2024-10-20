# Teeth Segmentation for Caries Detection

## Introduction

This project focuses on image segmentation using the U-Net architecture to detect tooth caries from dental images. Image segmentation plays a crucial role in computer vision by allowing us to classify each pixel into specific categories, enhancing our understanding of the image's structure. Unlike traditional image classification, which assigns a single label to the entire image, segmentation provides pixel-level labels, making it essential for applications in medical imaging and diagnostics.

The U-Net architecture, originally developed for biomedical image segmentation, has proven to be highly effective in producing high-resolution and accurate segmentation results. This project aims to leverage this architecture to achieve optimal results in detecting tooth caries.

## Dataset

The dataset used in this project consists of approximately 700 samples of 512x512 grayscale images of teeth, along with their corresponding segmentation masks. Each mask identifies the regions affected by caries, enabling precise training of the segmentation model.

## Problem Statement

The main goal is to train a segmentation model that accurately detects tooth caries while optimizing the model's performance and adhering to size limitations. Key aspects of this project include:

- Implementing effective data augmentation techniques to enhance the model's robustness.
- Selecting appropriate loss functions to improve segmentation accuracy.
- Identifying and resolving potential issues within the dataset, such as data poisoning, prior to training the model.

## Evaluation Metrics

The performance of the segmentation model will be evaluated using the **Dice score**, a widely recognized metric for segmentation tasks. The Dice score quantifies the overlap between predicted and actual segmentations, providing a reliable measure of the model's accuracy.

## Implementation Details

- **Frameworks Used**: This project is implemented using Python with TensorFlow/Keras for model training and evaluation.
- **Data Augmentation**: Techniques such as rotation, flipping, and scaling were employed to increase the variability of the training data.
- **Model Architecture**: The U-Net architecture was chosen for its ability to capture both contextual information and fine details in images.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/teeth-segmentation.git
   cd teeth-segmentation
2. Install requirements: 
   ```bash
   pip install -r requirements.txt
3. Run the python file:
   ```bash
   python run.py

