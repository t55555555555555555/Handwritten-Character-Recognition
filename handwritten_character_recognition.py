import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

# -------------------------------------------------------------------
# 1. Dataset Loading and Image Preprocessing
# -------------------------------------------------------------------

def load_and_preprocess_data():
    """
    Loads handwritten images, normalizes pixel intensities to [0, 1],
    and reshapes data to include the single grayscale channel (28, 28, 1).
    """
    print("[1/4] Loading handwritten dataset...")
    # Using the standard handwritten digits dataset (0-9)
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values from range [0, 255] to [0.0, 1.0]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Expand dimensions for CNN channel format: (num_samples, 28, 28, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, y_train, X_test, y_test


# -------------------------------------------------------------------
# 2. CNN Architecture Definition
# -------------------------------------------------------------------

def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Constructs a Convolutional Neural Network (CNN) for character recognition:
    - Conv2D layers for spatial pattern and edge detection
    - MaxPooling2D for downsampling
    - BatchNormalization and Dropout to prevent overfitting
    - Dense Softmax output layer for multi-class classification
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Second Convolutional Block
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Flatten & Dense Classification Head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -------------------------------------------------------------------
# 3. Model Training & Evaluation
# -------------------------------------------------------------------

def main():
    # Step 1: Prepare data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    num_classes = len(np.unique(y_train))
    class_labels = [str(i) for i in range(num_classes)]

    # Step 2: Build model
    print("\n[2/4] Building CNN architecture...")
    model = build_cnn_model(input_shape=(28, 28, 1), num_classes=num_classes)
    model.summary()

    # Step 3: Train model
    print("\n[3/4] Training the model...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=64,
        verbose=1,
    )

    # Step 4: Evaluate performance
    print("\n[4/4] Evaluating on test dataset...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Loss    : {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%\n")

    # Generate predictions and metric reports
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_labels))

    # Single-sample inference test
    sample_index = 0
    sample_image = X_test[sample_index:sample_index + 1]
    prediction = np.argmax(model.predict(sample_image))
    actual = y_test[sample_index]
    print(f"\nSample Prediction Test -> Predicted: '{prediction}', Actual: '{actual}'")

    # Save trained model to disk
    model.save("handwritten_character_cnn.h5")
    print("Model saved successfully as 'handwritten_character_cnn.h5'.")


if __name__ == "__main__":
    main()