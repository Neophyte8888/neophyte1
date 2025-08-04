import tensorflow as tf
from tensorflow.keras import layers, models

def create_simple_cnn(input_shape=(200, 200, 3)):
    """
    Creates and returns a simple Convolutional Neural Network model.

    Args:
        input_shape (tuple): The shape of the input images.

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    model = models.Sequential()

    # --- Convolutional Base ---
    # Layer 1: Convolution, Batch Norm, Pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Layer 2: Convolution, Batch Norm, Pooling
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Layer 3: Convolution, Batch Norm, Pooling
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Classifier Head ---
    # Flatten the feature map
    model.add(layers.Flatten())

    # Dense layer
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())

    # Dropout layer to reduce overfitting
    model.add(layers.Dropout(0.5))

    # Output layer for binary classification
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

if __name__ == '__main__':
    # Create the model
    model = create_simple_cnn()

    # Print a summary of the model architecture
    model.summary()

    # Compile the model to see configuration
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("\nModel created and compiled successfully.")
    print("This script can be imported to get the model architecture.")
