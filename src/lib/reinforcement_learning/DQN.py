import tensorflow as tf



class ConvDQN_CNNV2(tf.keras.Model):
    def __init__(self, action_dim=4):
        super().__init__()
        # Increased Conv-Layers to balance VRAM usage and performance
        self.conv1 = tf.keras.layers.Conv2D(16, 3, strides=1, activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(32, 3, strides=1, activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")

        # Flatten the output of the last convolutional layer
        self.flatten = tf.keras.layers.Flatten()

        # Increased Fully Connected Layers
        self.fc1 = tf.keras.layers.Dense(256, activation="relu")
        self.fc2 = tf.keras.layers.Dense(128, activation="relu")
        self.fc3 = tf.keras.layers.Dense(64, activation="relu")

        # Output layer for classification with action_dim classes and softmax activation
        self.output_layer = tf.keras.layers.Dense(action_dim, activation="softmax")

    def call(self, x):
        # x: (batch_size, height, width, channels)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten the output of the last convolutional layer
        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # Output layer for classification
        x = self.output_layer(x)

        return x  # Output is the index of the maximum value in the output vector