# Pre-trained Image Classifier to Identify Dog Breeds

A Python-based command-line tool built as part of the Udacity AI Programming with Python Nanodegree. This project uses pre-trained Convolutional Neural Network (CNN) architectures (**ResNet**, **AlexNet**, and **VGG**) to classify images, determine whether the subject is a dog or non-dog, and measure breed identification accuracy across different models.

---

## Overview

The primary goal of this project is to evaluate and compare three deep learning architectures trained on ImageNet. The program:

* Reads pet image filenames to extract ground-truth labels.
* Passes images through a pre-trained CNN classifier.
* Maps classifier labels against a reference file of dog names (`dognames.txt`).
* Calculates statistics on model accuracy for **dog classification**, **non-dog classification**, and **breed identification**.
* Prints a summary report comparing performance metrics across architectures.

---

## Project Architecture & Pipeline

1. **`check_images.py`**: Main driver script that coordinates the full workflow.
2. **`get_input_args.py`**: Parses command-line flags (`--dir`, `--arch`, `--dogfile`).
3. **`get_pet_labels.py`**: Extracts ground-truth pet labels from image file names.
4. **`classify_images.py`**: Runs images through the specified pre-trained CNN model using PyTorch/Torchvision.
5. **`adjust_results4_isadog.py`**: Evaluates whether real labels and predicted classifier labels are dogs or non-dogs using `dognames.txt`. Handles comma-separated multi-label predictions.
6. **`calculates_results_stats.py`**: Computes counts and percentages for model accuracy (% correct dogs, % correct non-dogs, % correct breeds).
7. **`print_results.py`**: Outputs execution runtime, summary statistics, misclassified dogs, and misclassified breeds.

---

## Performance Summary

Across a test dataset of 40 images (30 dogs, 10 non-dogs), the models achieved the following benchmark scores:

| Metric | ResNet | AlexNet | VGG |
| --- | --- | --- | --- |
| **% Correct Dogs** | 100.0% | 100.0% | 100.0% |
| **% Correct Non-Dogs** | 90.0% | 100.0% | 100.0% |
| **% Correct Breed** | 90.0% | 80.0% | 93.3% |

* **VGG** achieved the highest overall breed identification accuracy (93.3%).
* **ResNet** misclassified one non-dog image (`cat_01.jpg` predicted as a `norwegian elkhound`).
* **AlexNet** ran the fastest but had lower breed identification precision compared to VGG and ResNet.

---

## Installation & Requirements

### Dependencies

Ensure you have Python 3.x and the required libraries installed:

```bash
pip install torch torchvision torchvision-models pillow

```

---

## Usage

Run the classifier script from the terminal by passing the image directory, architecture, and dog names reference file:

### Single Model Runs

```bash
# Run with ResNet
python check_images.py --dir pet_images/ --arch resnet --dogfile dognames.txt

# Run with AlexNet
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt

# Run with VGG
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt

```

### Batch Run (All Models)

To execute all models sequentially and write output logs to text files:

```bash
sh run_models_batch.sh

```

---

## Command Line Arguments

* `--dir` : Path to the folder of pet images (default: `pet_images/`)
* `--arch` : CNN model architecture to use (`resnet`, `alexnet`, or `vgg`; default: `vgg`)
* `--dogfile` : Text file containing all valid dog names (default: `dognames.txt`)
