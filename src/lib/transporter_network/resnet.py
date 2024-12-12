import tensorflow as tf


# TODO: check with dilation?
class IdentityBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size=(3, 3), filters=64, include_batchnorm=False, activation=True,
                 padding_type='same', output_depth=None, **kwargs):
        super(IdentityBlock, self).__init__(**kwargs)
        self.padding_type = padding_type
        self.kernel_size = kernel_size
        self.filters = filters
        self.output_depth = output_depth if output_depth is not None else filters
        self.include_batchnorm = include_batchnorm
        self.activation = activation
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1),
                                    strides=(1, 1), padding=self.padding_type)
        if self.include_batchnorm:
            self.b_1 = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv_2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size,
                             strides=(1, 1), padding=self.padding_type)
        if self.include_batchnorm:
            self.b_2 = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv_3 = tf.keras.layers.Conv2D(filters=self.output_depth, kernel_size=(1, 1),
                             strides=(1, 1), padding=self.padding_type)
        if self.include_batchnorm:
            self.b_3 = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs, **kwargs):
        output = self.conv_1(inputs)
        if self.include_batchnorm:
            output = self.b_1(output)
        output = tf.keras.layers.ReLU()(output)

        output = self.conv_2(output)
        if self.include_batchnorm:
            output = self.b_2(output)
        output = tf.keras.layers.ReLU()(output)

        output = self.conv_3(output)
        if self.include_batchnorm:
            output = self.b_3(output)
        output = tf.keras.layers.ReLU()(output)

        output = tf.keras.layers.add([inputs, output])
        if self.activation:
            output = tf.keras.layers.ReLU()(output)
        return output


# TODO: check with dilation --> there is no stride then ....
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size=(3, 3), filters=64, stride=(2, 2), include_batchnorm=False, activation=True,
                 padding_type='same', output_depth=None, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.padding_type = padding_type
        self.kernel_size = kernel_size
        self.filters = filters
        self.output_depth = output_depth if output_depth is not None else filters
        self.stride = stride
        self.include_batchnorm = include_batchnorm
        self.activation = activation
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1),
                             strides=self.stride, padding=self.padding_type)
        if self.include_batchnorm:
            self.b_1 = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv_2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size,
                             strides=(1, 1), padding=self.padding_type)
        if self.include_batchnorm:
            self.b_2 = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv_3 = tf.keras.layers.Conv2D(filters=self.output_depth, kernel_size=(1, 1),
                             strides=(1, 1), padding=self.padding_type)
        if self.include_batchnorm:
            self.b_3 = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv_skip = tf.keras.layers.Conv2D(filters=self.output_depth, kernel_size=(1, 1),
                                strides=self.stride, padding=self.padding_type)
        if self.include_batchnorm:
            self.b_skip = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs, **kwargs):
        output = self.conv_1(inputs)
        if self.include_batchnorm:
            output = self.b_1(output)
        output = tf.keras.layers.ReLU()(output)

        output = self.conv_2(output)
        if self.include_batchnorm:
            output = self.b_2(output)
        output = tf.keras.layers.ReLU()(output)

        output = self.conv_3(output)
        if self.include_batchnorm:
            output = self.b_3(output)

        skip = self.conv_skip(inputs)
        if self.include_batchnorm:
            output = self.b_skip(output)
        output = tf.keras.layers.add([skip, output])
        if self.activation:
            output = tf.keras.layers.ReLU()(output)
        return output


class ResNetEncoder(tf.keras.layers.Layer):
    def __init__(self, kernel_size=(3, 3), include_batchnorm=False, padding_type='same', **kwargs):
        super(ResNetEncoder, self).__init__(**kwargs)
        self.padding_type = padding_type
        self.kernel_size = kernel_size
        self.include_batchnorm = include_batchnorm

        self.conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=self.kernel_size, strides=(1, 1), padding=self.padding_type)
        if self.include_batchnorm:
            self.b_1 = tf.keras.layers.BatchNormalization(axis=-1)

        self.cb_2 = ConvBlock(self.kernel_size, 64, (1, 1), self.include_batchnorm, padding_type=self.padding_type)
        self.id_2 = IdentityBlock(self.kernel_size, 64, self.include_batchnorm, padding_type=self.padding_type)

        self.cb_3 = ConvBlock(self.kernel_size, 128, (2, 2), self.include_batchnorm, padding_type=self.padding_type)
        self.id_3 = IdentityBlock(self.kernel_size, 128, self.include_batchnorm, padding_type=self.padding_type)

        self.cb_4 = ConvBlock(self.kernel_size, 256, (2, 2), self.include_batchnorm, padding_type=self.padding_type)
        self.id_4 = IdentityBlock(self.kernel_size, 256, self.include_batchnorm, padding_type=self.padding_type)

        self.cb_5 = ConvBlock(self.kernel_size, 512, (2, 2), self.include_batchnorm, padding_type=self.padding_type)
        self.id_5 = IdentityBlock(self.kernel_size, 512, self.include_batchnorm, padding_type=self.padding_type)

    def call(self, inputs, **kwargs):
        output = self.conv_1(inputs)
        if self.include_batchnorm:
            output = self.b_1(output)
        output = tf.keras.layers.ReLU()(output)

        output = self.cb_2(output)
        output = self.id_2(output)

        output = self.cb_3(output)
        output = self.id_3(output)

        output = self.cb_4(output)
        output = self.id_4(output)

        output = self.cb_5(output)
        output = self.id_5(output)
        return output


class ResNetDecoder(tf.keras.layers.Layer):
    def __init__(self, kernel_size=(3, 3), output_depth=1, include_batchnorm=False, padding_type='same', **kwargs):
        super(ResNetDecoder, self).__init__(**kwargs)
        self.padding_type = padding_type
        self.kernel_size = kernel_size
        self.include_batchnorm = include_batchnorm
        self.output_depth = output_depth

        self.cb_6 = ConvBlock(self.kernel_size, 256, (1, 1), self.include_batchnorm, padding_type=self.padding_type)
        self.id_6 = IdentityBlock(self.kernel_size, 256, self.include_batchnorm, padding_type=self.padding_type)

        self.cb_7 = ConvBlock(self.kernel_size, 128, (1, 1), self.include_batchnorm, padding_type=self.padding_type)
        self.id_7 = IdentityBlock(self.kernel_size, 128, self.include_batchnorm, padding_type=self.padding_type)

        self.cb_8 = ConvBlock(self.kernel_size, 64, (1, 1), self.include_batchnorm, padding_type=self.padding_type)
        self.id_8 = IdentityBlock(self.kernel_size, 64, self.include_batchnorm, padding_type=self.padding_type)

        self.cb_9 = ConvBlock(self.kernel_size, 64, (1, 1), self.include_batchnorm, False,
                              padding_type=self.padding_type, output_depth=self.output_depth)
        self.id_9 = IdentityBlock(self.kernel_size, 64, self.include_batchnorm, False,
                                  padding_type=self.padding_type, output_depth=self.output_depth)

    def call(self, inputs, **kwargs):
        output = self.cb_6(inputs)
        output = self.id_6(output)

        output = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(output)

        output = self.cb_7(output)
        output = self.id_7(output)

        output = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(output)

        output = self.cb_8(output)
        output = self.id_8(output)

        output = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(output)

        output = self.cb_9(output)
        output = self.id_9(output)

        return output


class ResNet(tf.keras.layers.Layer):
    def __init__(self, kernel_size=(3, 3), output_depth=1, include_batchnorm=False, padding_type='same', **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.padding_type = padding_type
        self.kernel_size = kernel_size
        self.include_batchnorm = include_batchnorm
        self.output_depth = output_depth

        self.encoder = ResNetEncoder(kernel_size=self.kernel_size, include_batchnorm=self.include_batchnorm,
                                     padding_type=self.padding_type,
                                     **kwargs)
        self.decoder = ResNetDecoder(kernel_size=self.kernel_size, output_depth=self.output_depth,
                                     include_batchnorm=self.include_batchnorm,
                                     padding_type=self.padding_type,
                                     **kwargs)

    def call(self, inputs, **kwargs):
        output = self.encoder(inputs)
        output = self.decoder(output)
        return output