import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


# The model was trained with 224 x 224 images, so prediction images
# must be resized to the same size.
IMG_SIZE = (224, 224)


def parse_args():
    # This script is run from the terminal.
    # Example:
    # python predict_single_image.py data/test/healthy/example.jpg
    # image_path is required. The model paths have default values.
    parser = argparse.ArgumentParser(description="Predict healthy or bleached coral from one image.")
    parser.add_argument("image_path", help="Image path, for example data/test/healthy/example.jpg")
    parser.add_argument("--model-path", default="outputs/best_mobilenetv2.keras")
    parser.add_argument("--class-names-path", default="outputs/class_names.json")
    return parser.parse_args()


def main():
    # Read the image path and optional model paths from the command line.
    args = parse_args()

    # Path objects make file paths easier and safer to work with.
    image_path = Path(args.image_path)
    model_path = Path(args.model_path)
    class_names_path = Path(args.class_names_path)

    # Stop early with a clear message if an important file is missing.
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train the model first.")
    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names not found: {class_names_path}. Train the model first.")

    # class_names.json stores the label order used during training.
    # Example: ["bleached", "healthy"]
    # This order matters because the model outputs probabilities in the same order.
    with class_names_path.open("r", encoding="utf-8") as f:
        class_names = json.load(f)

    # Load the trained model saved by train_coral_cnn.py.
    model = tf.keras.models.load_model(model_path)

    # Load the image, resize it to 224 x 224, and convert it into numbers.
    # Neural networks do not read images as pictures; they read arrays of pixel values.
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)

    # Add one extra dimension because the model expects a batch of images.
    # Shape changes from (224, 224, 3) to (1, 224, 224, 3).
    img_array = np.expand_dims(img_array, axis=0)

    # The model returns one probability for each class.
    # Example: [0.82, 0.18] means 82% for class 0 and 18% for class 1.
    probabilities = model.predict(img_array, verbose=0)[0]

    # np.argmax finds the index of the largest probability.
    predicted_index = int(np.argmax(probabilities))

    # Print the final predicted class and the confidence score.
    print("Predicted class:", class_names[predicted_index])
    print("Confidence:", round(float(probabilities[predicted_index]), 4))
    print("\nAll probabilities:")

    # Print every class probability so the result is transparent.
    for class_name, probability in zip(class_names, probabilities):
        print(f"{class_name}: {probability:.4f}")


if __name__ == "__main__":
    # This means: only run main() when this file is executed directly.
    # It will not automatically run if another file imports it.
    main()
