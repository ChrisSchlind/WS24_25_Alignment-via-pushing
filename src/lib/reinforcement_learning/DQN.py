import tensorflow as tf
from .resnet import ResNet

from loguru import logger

class ConvDQN_ResNet(tf.keras.Model):
    def __init__(self, initializer=tf.keras.initializers.GlorotUniform()):
        super().__init__()
        # Initializer for the weights
        self.initializer = initializer

        # ResNet Block 1: 
        self.resnet_block_1 = ResNet(kernel_size=(3, 3), output_depth=64, include_batchnorm=True)

        # ResNet Block 2: 
        self.resnet_block_2 = ResNet(kernel_size=(3, 3), output_depth=256, include_batchnorm=True)        

        # Conv2D layer for the heatmap output (H, W, 1)
        self.heatmap = tf.keras.layers.Conv2D(1, 3, strides=1, padding='same', kernel_initializer=self.initializer)

    def call(self, inputs):
        # Input shape: (batch_size, 88, 88, 7)

        logger.debug(f"Input shape: {inputs.shape}")

        # First ResNet-Block
        x = self.resnet_block_1(inputs)

        # Second ResNet-Block
        x = self.resnet_block_2(x)
     
        # Final heatmap (H, W, 1)
        x = self.heatmap(x)

        # Delete the last 2 rows and columns to get the correct heatmap size and remove ResNet errors at the edges
        x = tf.keras.layers.Cropping2D(cropping=((2, 2), (2, 2)))(x)

        logger.debug(f"Output shape: {x.shape}")

        # Return the heatmap (no activation because this is a continuous value map)
        return x  # Heatmap of dimension (batch_size, 84, 84, 1)


class ConvDQN_FCNV2(tf.keras.Model):
    def __init__(self, initializer=tf.keras.initializers.GlorotUniform()):
        super().__init__()
        # Initializer for the weights
        self.initializer = initializer

        # 4 Conv-Layers, each with 'same' padding to keep the output size the same as the input size
        self.conv1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation="relu", kernel_initializer=self.initializer)
        self.conv2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation="relu", kernel_initializer=self.initializer)
        self.conv3 = tf.keras.layers.Conv2D(128, 3, strides=1, padding='same', activation="relu", kernel_initializer=self.initializer)
        self.conv4 = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', activation="relu", kernel_initializer=self.initializer)

        # Attention Map (Sigmoid to normalize the values between 0 and 1)
        self.attention_map = tf.keras.layers.Conv2D(1, 3, strides=1, padding='same', activation="sigmoid", kernel_initializer=self.initializer)

        # Final Conv2D layer for the heatmap output (H, W, 1)
        self.heatmap = tf.keras.layers.Conv2D(1, 3, strides=1, padding='same', activation=None, kernel_initializer=self.initializer)

    def call(self, inputs):
        # Input shape: (batch_size, 88, 88, 7)

        logger.debug(f"Input shape: {inputs.shape}")

        # 4 Conv-Layers
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Generate attention map
        attention = self.attention_map(x)

        # Apply attention to the last convolutional layer's output
        x = x * attention

        # Final heatmap
        x = self.heatmap(x)

        # Crop the heatmap to the desired size of 84x84
        x = tf.keras.layers.Cropping2D(cropping=((2, 2), (2, 2)))(x)

        logger.debug(f"Output shape: {x.shape}")

        # Return the heatmap (no activation because this is a continuous value map)
        return x  # Heatmap of dimension (batch_size, 84, 84, 1)


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

    def call(self, inputs):
        # Input shape: (batch_size, 84, 84, 7)

        logger.debug(f"Input shape: {inputs.shape}")

        # Convolutional layers
        x = self.conv1(inputs)
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

        logger.debug(f"Output shape: {x.shape}")

        return x  # Output is the index of the maximum value in the output vector