import tensorflow as tf
import tensorflow_addons as tfa
from einops import repeat, rearrange
import numpy as np

from .resnet import ResNet


class TransporterNetworkAttention(tf.keras.Model):
    def __init__(self, crop_size, n_orentation_bins, **kwargs):
        super().__init__(**kwargs)
        self.crop_size = crop_size
        self.n_orentation_bins = n_orentation_bins
        rotation_steps = np.linspace(0, np.pi, n_orentation_bins)
        self.rotations = tf.convert_to_tensor(rotation_steps, dtype=tf.float32)

        self.flatten = tf.keras.layers.Flatten()

        self.tn_attention = ResNet(kernel_size=(3, 3), output_depth=1, include_batchnorm=False, padding_type='same')
        self.yaw_classification = tf.keras.layers.Conv2D(1, (crop_size, crop_size), padding='valid')

        self.metrics_attentions = tf.keras.metrics.Mean(name='loss_attentions')
        self.metrics_yaw = tf.keras.metrics.Mean(name='loss_yaw')

    def infer(self, inputs, training=False):
        attentions = self.tn_attention(inputs)
        
        crop_mids = self.get_max_locations(attentions)
        translated_inputs = self.translate_images(inputs, crop_mids)
        
        rotated_images = self.repeat_and_rotate_images(translated_inputs)
        crops = self.mid_crop(rotated_images)
        yaws = self.yaw_classification(crops)
        
        attentions = tf.squeeze(attentions, axis=-1)
        yaws = tf.squeeze(yaws, axis=[-3, -2, -1])

        f_attentions = self.flatten(attentions)
        f_attentions = tf.nn.softmax(f_attentions)
        attentions = tf.reshape(f_attentions, tf.shape(attentions))

        yaws = tf.nn.softmax(yaws)
        return attentions, yaws
    
    def apply_softmax_to_outputs(self, attentions, rotations):
        s_attentions = self.flatten(attentions)
        s_attentions = tf.nn.softmax(s_attentions)
        attentions = tf.reshape(s_attentions, tf.shape(attentions))
        rotations = tf.nn.softmax(rotations)
        return attentions, rotations
    
    def call(self, inputs):
        attentions = self.tn_attention(inputs[0])
        yaws = self.yaw_classification(inputs[1])

        attentions = tf.squeeze(attentions, axis=-1)
        yaws = tf.squeeze(yaws, axis=[-3, -2, -1])
        return attentions, yaws
    
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            attentions, yaws = self(inputs)
            logits_attentions = self.flatten(attentions)
            labels_attentions = self.flatten(targets[0])
            loss_attentions = self.loss(labels_attentions, logits_attentions)
            
            logits_yaws = self.flatten(yaws)
            labels_yaws = self.flatten(targets[1])
            loss_yaws = self.loss(labels_yaws, logits_yaws)
            
            loss = loss_attentions + loss_yaws
        
        gradients_attentions, gradients_yaws = tape.gradient(loss, [self.tn_attention.trainable_variables, self.yaw_classification.trainable_variables])
        self.optimizer[0].apply_gradients(zip(gradients_attentions, self.tn_attention.trainable_variables))
        self.optimizer[1].apply_gradients(zip(gradients_yaws, self.yaw_classification.trainable_variables))
        
        self.metrics_attentions.update_state(loss_attentions)
        self.metrics_yaw.update_state(loss_yaws)
        return {"loss_attentions": self.metrics_attentions.result(),
                "loss_yaws": self.metrics_yaw.result()}
    
    @staticmethod
    def translate_images(images, new_centers):
        image_mid_point = tf.cast(tf.shape(images)[1:3], dtype=tf.float32) / 2.0
        image_mid_point = repeat(image_mid_point, 's -> b s', b=tf.shape(images)[0])
        translations = image_mid_point - tf.cast(new_centers, dtype=tf.float32)
        translations = tf.roll(translations, 1, axis=1)
        translated_images = tfa.image.translate(images, translations)
        return translated_images


    def repeat_and_rotate_images(self, images):
        repeated_images = repeat(images, 'b h w c -> (b n) h w c', n=self.n_orentation_bins)
        rotations = repeat(self.rotations, 'n -> (b n)', b=tf.shape(images)[0])

        rotated_images = tfa.image.rotate(repeated_images, rotations, interpolation='bilinear')
        
        rotated_images = rearrange(rotated_images, '(b n) h w c -> b n h w c', n=self.n_orentation_bins)

        return rotated_images


    def get_max_locations(self, inputs):
        inputs_flat = self.flatten(inputs)
        max_locations = tf.unravel_index(tf.math.argmax(inputs_flat, axis=-1),
                                        tf.shape(inputs, out_type=tf.int64)[1:])
        max_locations = tf.transpose(max_locations)[..., :2]
        return max_locations


    def mid_crop(self, inputs):
        img_shape = tf.shape(inputs)[-3:-1]
        crop_start = tf.cast((img_shape - self.crop_size) / 2, dtype=tf.int32)
        crop_end = tf.cast(crop_start + self.crop_size, dtype=tf.int32)
        cropped_inputs = inputs[..., crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], :]
        return cropped_inputs