import csv  # Saves CNN prediction results into a .csv file.
import json  # Saves class names, for example ["bleached", "healthy"].
from pathlib import Path  # Handles file paths, for example DATA_DIR / "train".

import numpy as np  # Finds the class with the highest probability using np.argmax().
import tensorflow as tf  # Loads images, builds the CNN, and trains the model.


DATA_DIR = Path("data")  # Dataset folder.
OUTPUT_DIR = Path("outputs")  # Folder for saved model and prediction outputs.
IMG_SIZE = (224, 224)  # All images are resized to 224 x 224 pixels.
BATCH_SIZE = 16  # The model processes 16 images at a time.
EPOCHS = 15  # Maximum number of training rounds.


OUTPUT_DIR.mkdir(exist_ok=True)  # Create the output folder if it does not exist.

# Load the training dataset.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",  # Read images from data/train.
    image_size=IMG_SIZE,  # Resize images to 224 x 224.
    batch_size=BATCH_SIZE,  # Use 16 images per batch.
    label_mode="int",  # Store labels as integers, such as 0 and 1.
    shuffle=True,  # Shuffle training images to help the model learn better.
)

# Load the validation dataset.
# The validation set checks model performance during training.
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

# Load the test dataset.
# In Nick's part, this is used to export CNN predictions for later evaluation.
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

# Save the class names so the prediction script knows the label order.
class_names = train_ds.class_names
with (OUTPUT_DIR / "class_names.json").open("w", encoding="utf-8") as file:
    json.dump(class_names, file)

# Prefetch prepares the next image batch while the current batch is being used.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Load MobileNetV2 as the base CNN model.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),  # Input shape is 224 x 224 x 3 because images are RGB.
    include_top=False,  # Remove MobileNetV2's original classification layer.
    weights="imagenet",  # Use weights pretrained on ImageNet.
)
base_model.trainable = False  # Freeze MobileNetV2's original parameters.

# Build the full CNN model.
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=IMG_SIZE + (3,)),  # Define the input image size.
        tf.keras.layers.RandomFlip("horizontal"),  # Randomly flip images horizontally.
        tf.keras.layers.RandomRotation(0.10),  # Randomly rotate images.
        tf.keras.layers.RandomZoom(0.10),  # Randomly zoom images.
        tf.keras.layers.RandomContrast(0.10),  # Randomly change contrast.
        tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input),  # Format pixels for MobileNetV2.
        base_model,  # Use MobileNetV2 to extract image features.
        tf.keras.layers.GlobalAveragePooling2D(),  # Convert feature maps into one feature vector.
        tf.keras.layers.Dropout(0.30),  # Drop 30% of features during training to reduce overfitting.
        tf.keras.layers.Dense(len(class_names), activation="softmax"),  # Output class probabilities.
    ]
)

# Compile the model. This tells Keras how to train it.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Adam updates the model parameters.
    loss="sparse_categorical_crossentropy",  # Loss function for integer-labelled classification.
    metrics=["accuracy"],  # Show accuracy during training.
)

# Train the model and save the best version.
checkpoint_path = OUTPUT_DIR / "best_mobilenetv2.keras"
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ],
)

# Load the best model saved during training.
model = tf.keras.models.load_model(checkpoint_path)

# Export CNN prediction results to a CSV file.
# This file can be passed to the evaluation teammate for confusion matrix and performance metrics.
with (OUTPUT_DIR / "cnn_predictions.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["true_label", "predicted_label", "confidence"])

    for images, labels in test_ds:
        probabilities = model.predict(images, verbose=0)
        predicted_indexes = np.argmax(probabilities, axis=1)

        for true_index, predicted_index, probability_row in zip(labels.numpy(), predicted_indexes, probabilities):
            writer.writerow(
                [
                    class_names[int(true_index)],
                    class_names[int(predicted_index)],
                    float(probability_row[int(predicted_index)]),
                ]
            )

print("CNN model saved to:", checkpoint_path)
print("CNN predictions saved to:", OUTPUT_DIR / "cnn_predictions.csv")
