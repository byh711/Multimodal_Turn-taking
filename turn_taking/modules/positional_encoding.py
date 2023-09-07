# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional Encoding Layer for Transformer Architecture.
    This layer adds positional encodings to the input sequences.
    """
    
    def __init__(self, position, d_model):
        """
        Constructor for the PositionalEncoding layer.
        
        Parameters:
        - position: Maximum position encoding.
        - d_model: Model dimension.
        """
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        """
        Compute the angles for positional encoding.
        
        Parameters:
        - position: Tensor containing positions.
        - i: Tensor containing dimension indices.
        - d_model: Model dimension.
        
        Returns:
        - Tensor containing the computed angles.
        """
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        """
        Compute the positional encoding matrix.
        
        Parameters:
        - position: Maximum position encoding.
        - d_model: Model dimension.
        
        Returns:
        - Tensor containing the positional encoding matrix.
        """
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)

        # Apply sine function to even indices (2i) of the array
        sines = tf.math.sin(angle_rads[:, 0::2])

        # Apply cosine function to odd indices (2i+1) of the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)
    
    def get_config(self):
        config = super().get_config().copy()
        return config

    def call(self, inputs):
        """
        Add the positional encodings to the input sequences.
        
        Parameters:
        - inputs: Input sequences.
        
        Returns:
        - Tensor containing input sequences with added positional encodings.
        """
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
