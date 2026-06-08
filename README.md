# Star vs Galaxy Classification with CNN 

This project implements a binary image classification model to distinguish **stars** from **galaxies** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

The goal is to classify astronomical cutout images into two classes:

- **Class 0:** Star
- **Class 1:** Galaxy

The project also addresses class imbalance by using **class weights**, so the model can learn from all available images without reducing the larger class.

---

## Project Overview

The script performs the full machine learning workflow:

1. Loads star and galaxy images from local folders.
2. Converts images to grayscale.
3. Resizes all images to **64x64 pixels**.
4. Normalizes pixel values between **0 and 1**.
5. Splits the dataset into training and test sets.
6. Computes class weights to handle class imbalance.
7. Trains a CNN model for binary classification.
8. Evaluates the model using accuracy, loss, confusion matrix, and classification report.
9. Saves the trained model as an `.h5` file.

---


## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

---

## Model Architecture

The model is a simple CNN composed of:

- 3 convolutional layers
- 3 max pooling layers
- 1 flatten layer
- 1 dense hidden layer
- 1 dropout layer
- 1 sigmoid output neuron

The final layer uses a **sigmoid activation function**, which returns a probability between 0 and 1:

- values close to **0** indicate **Star**
- values close to **1** indicate **Galaxy**

---

## Training Configuration

| Parameter | Value |
|---|---:|
| Image size | 64x64 |
| Color mode | Grayscale |
| Epochs | 10 |
| Batch size | 32 |
| Train-test split | 80% / 20% |
| Optimizer | Adam |
| Loss function | Binary Crossentropy |
| Random seed | 42 |

---

## Results

The final evaluation on the test set produced the following results:

| Metric | Value |
|---|---:|
| Test loss | 0.3058 |
| Test accuracy | 85.34% |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Star | 0.95 | 0.86 | 0.90 | 609 |
| Galaxy | 0.65 | 0.84 | 0.73 | 189 |
| Accuracy |  |  | 0.85 | 798 |
| Macro avg | 0.80 | 0.85 | 0.82 | 798 |
| Weighted avg | 0.87 | 0.85 | 0.86 | 798 |

---

## Confusion Matrix

| Actual / Predicted | Star | Galaxy |
|---|---:|---:|
| Star | 522 | 87 |
| Galaxy | 30 | 159 |

The model correctly classified:

- **522 stars** as stars
- **159 galaxies** as galaxies

The model misclassified:

- **87 stars** as galaxies
- **30 galaxies** as stars

Overall, the model achieved a good test accuracy of about **85%**. It performs particularly well on the **Star** class, with high precision and strong F1-score. For the **Galaxy** class, recall is good, meaning that most galaxies are detected, but precision is lower because some stars are incorrectly classified as galaxies.

---


## How to Run the Project

Install the required libraries:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

Run the script:

```bash
python star_galaxy_weighted.py
```

After training, the script saves:

```text
star_galaxy_weighted_model.h5
```

and stores the result plots inside the `results/` folder.

---

## Output Files

| File | Description |
|---|---|
| `training_accuracy.png` | Training and test accuracy across epochs |
| `training_loss.png` | Training and test loss across epochs |
| `confusion_matrix.png` | Confusion matrix on the test set |
| `classification_report.txt` | Precision, recall, F1-score, and support |
| `star_galaxy_weighted_model.h5` | Saved trained model |



## Conclusion

This project shows how a CNN can be used to classify astronomical images into stars and galaxies. The use of class weights allows the model to handle an imbalanced dataset without discarding useful images. The final model achieved an accuracy of approximately **85%**, with strong performance on star detection and good recall for galaxies.
