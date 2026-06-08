"""
Star vs Galaxy Classification with Class Weights

Goal:
- Class 0 = Star
- Class 1 = Galaxy

The dataset is imbalanced:
- about 900 star images
- about 3000 galaxy images

"""

import os
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# ==============================
# BASIC SETTINGS
# ==============================


DATASET_PATH = "dataset/Cutout Files"
STAR_PATH = r"C:\Users\Utente\Desktop\Cutout Files\star"
GALAXY_PATH = r"C:\Users\Utente\Desktop\Cutout Files\galaxy"

IMG_SIZE = 64
EPOCHS = 10
BATCH_SIZE = 32
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ==============================
# FUNCTIONS
# ==============================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


#Checks if the star and galaxy folders exist
def check_folders():
    if not os.path.exists(STAR_PATH):
        raise FileNotFoundError(f"Folder not found: {STAR_PATH}")

    if not os.path.exists(GALAXY_PATH):
        raise FileNotFoundError(f"Folder not found: {GALAXY_PATH}")

    print("Folders found correctly.")
    print("Star images:", len(get_image_files(STAR_PATH)))
    print("Galaxy images:", len(get_image_files(GALAXY_PATH)))


#Returns only image files from a folder
def get_image_files(folder):
    files = []

    for filename in os.listdir(folder):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            files.append(filename)

    return files


def load_images(folder, label):
    images = []
    labels = []

    filenames = get_image_files(folder)
    random.shuffle(filenames)

    for filename in filenames:
        image_path = os.path.join(folder, filename)

        try:
            img = keras.preprocessing.image.load_img(
                image_path,
                target_size=(IMG_SIZE, IMG_SIZE),
                color_mode="grayscale"
            )

            img_array = keras.preprocessing.image.img_to_array(img)

            # Normalize pixel values between 0 and 1
            img_array = img_array / 255.0

            images.append(img_array)
            labels.append(label)

        except:
            pass

    return images, labels


#Computes class weights manually
def compute_class_weights(y_train):
    total = len(y_train)

    count_star = np.sum(y_train == 0)
    count_galaxy = np.sum(y_train == 1)

    weight_star = total / (2 * count_star)
    weight_galaxy = total / (2 * count_galaxy)

    class_weights = {
        0: weight_star,
        1: weight_galaxy
    }

    return class_weights


#Show some examples images 
def show_examples(X, y):
    class_names = ["Star", "Galaxy"]

    plt.figure(figsize=(10, 5))

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[i].squeeze(), cmap="gray")
        plt.title(class_names[y[i]])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

#CNN model
def create_model():
    model = keras.Sequential([
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(IMG_SIZE, IMG_SIZE, 1)
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


#Plot accuracy and loss
def plot_training(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(RESULTS_DIR, "training_accuracy.png")
    plt.savefig(acc_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(RESULTS_DIR, "training_loss.png")
    plt.savefig(loss_path)
    plt.close()


#Plot the confusion matrix 
def plot_confusion_matrix(y_test, y_pred):
    class_names = ["Star", "Galaxy"]

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

#Predict one single image after training 
def predict_single_image(model, image_path):
    img = keras.preprocessing.image.load_img(
        image_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale"
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "Galaxy"
    else:
        label = "Star"

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(f"Prediction: {label}")
    plt.show()

    print("Galaxy probability:", prediction)


# ==============================
# MAIN PROGRAM
# ==============================

def main():
    set_seed(SEED)

    print("Star vs Galaxy Classification")
    print("TensorFlow version:", tf.__version__)

    # 2. Check folders
    check_folders()

    # 3. Load all images
    star_images, star_labels = load_images(STAR_PATH, label=0)
    galaxy_images, galaxy_labels = load_images(GALAXY_PATH, label=1)

    print("\nLoaded images:")
    print("Stars:", len(star_images))
    print("Galaxies:", len(galaxy_images))

    # 4. Create X and y
    X = np.array(star_images + galaxy_images)
    y = np.array(star_labels + galaxy_labels)

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    # 5. Shuffle the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    # 6. Show some examples
    show_examples(X, y)

    # 7. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y #same class proportion in train and test 
    )

    print("\nTrain:", X_train.shape)
    print("Test:", X_test.shape)

    # 8. Compute class weights
    class_weights = compute_class_weights(y_train)

    print("\nClass weights:")
    print("Star weight:", class_weights[0])
    print("Galaxy weight:", class_weights[1])

    # 9. Create the model
    model = create_model()
    model.summary()

    # 10. Train the model using class weights
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        class_weight=class_weights
    )

    # 11. Plot training results
    plot_training(history)

    # 12. Final evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print("\nFinal results")
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)

    # 13. Predictions on test set
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

    # 14. Confusion matrix and classification report
    plot_confusion_matrix(y_test, y_pred)

    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Star", "Galaxy"]
    ))

    # Save classification report to a text file
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Star", "Galaxy"]
    )
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Final results\n")
        f.write(f"Test loss: {test_loss}\n")
        f.write(f"Test accuracy: {test_accuracy}\n\n")
        f.write(report)

    # 15. Save the model
    model.save("star_galaxy_weighted_model.h5")
    print("\nModel saved as star_galaxy_weighted_model.h5")

    # To test one single image after training: 
    # predict_single_image(model, "path/to/your/image.jpg")


if __name__ == "__main__":
    main()
