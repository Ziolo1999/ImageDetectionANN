# Image Detection with Artificial Neural Networks

This repository contains the code and resources for image detection tasks, which were developed as part of my Artificial Neural Networks (ANN) coursework. The project involves fine-tuning a model using the Scene dataset and implements two tasks: a supervised learning model and a self-supervised approach. The self-supervised approach includes models trained with rotated images and perturbations.

## Dataset
The Scene dataset is used for fine-tuning the models. It is a comprehensive collection of images depicting various scenes, which provides a diverse set of visual inputs to train and test the image detection models effectively.

## Models

The project focuses on image detection using neural networks. Two main tasks are addressed:
1. **Supervised Learning Model**: A model trained using labeled images from the Scene dataset.
2. **Self-Supervised Learning Approach**: This task involves two models:
   - **Model trained with rotated images**: This model augments the training data by rotating images.
   - **Model trained with perturbations**: This model applies various perturbations to the images to enhance robustness and generalization.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage


All models are trained and evaluated using scripts organized into separate folders for supervised learning and self-supervised learning.

### Supervised Learning Model

To train the supervised model, use the following command:

```bash
jupyter notebook supervised_task/report.ipynb
```

### Self-Supervised Learning Models

For the rotated images and perturbation models:

```bash
jupyter notebook self_supervised_task/report.ipynb
```

### Evaluation
The evaluation included accuracy comparisons, ScoreCAM visualizations, and model inversion comparisons. Accuracy measured the models' performance, while ScoreCAM visualizations highlighted the image regions influencing predictions. Model inversion comparisons assessed the quality and robustness of the learned features.
To evaluate the models, use the following commands:

```bash
jupyter notebook evaluation/evaluation.ipynb
```

## Results

Comprehensive details of the results, methodology, and analysis are documented in the PDF report titled `Artificial Neural Networks - Project.pdf`.
