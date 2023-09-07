import tensorflow as tf

def scaled_dot_product_attention(query, key, value, mask=None):
    """Computes the scaled dot product attention given query, key, and value matrices.
    
    Parameters:
    - query, key, value (tf.Tensor): Input matrices for attention computation.
    - mask (tf.Tensor, optional): Mask to apply on attention scores. Defaults to None.
    
    Returns:
    - output (tf.Tensor): The attention output.
    - attention_weights (tf.Tensor): The attention weights.
    """
    # Sizes:
    # Query: (batch_size, num_heads, length of query sentence, d_model/num_heads)
    # Key: (batch_size, num_heads, length of key sentence, d_model/num_heads)
    # Value: (batch_size, num_heads, length of value sentence, d_model/num_heads)
    # Padding mask: (batch_size, 1, 1, length of key sentence)

    # Matrix multiplication of Q and K to get the attention score matrix.
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    
    # Scaling by dividing by the square root of depth.
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    
    # If there's a mask, add very small negative values to masked positions. 
    # After passing through softmax, these positions will become nearly zero.
    if mask is not None:
        logits += (mask * -1e9)
    
    # Softmax is applied on the last dimension, which is the length of the key sentence.
    attention_weights = tf.nn.softmax(logits, axis=-1)
    
    # Compute the output using the attention weights
    output = tf.matmul(attention_weights, value)

    return output, attention_weights

class MultiHead_Attention(tf.keras.layers.Layer):
    """MultiHead Attention mechanism that uses multiple attention heads to capture different 
    types of information from the input.
    
    Attributes:
    - num_heads (int): Number of attention heads.
    - d_model (int): Depth of the model.
    - depth (int): Depth of each attention head.
    - query_dense, key_dense, value_dense (tf.keras.layers.Dense): Dense layers for query, key, and value.
    - dense (tf.keras.layers.Dense): Final dense layer.
    """
    
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHead_Attention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        # Depth is the value when d_model is divided by num_heads. 
        # Reference from the paper: 64
        self.depth = d_model // self.num_heads

        # Define dense layers for WQ, WK, WV
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        # Define the final dense layer, WO
        self.dense = tf.keras.layers.Dense(units=d_model)

    # Function to split q, k, v by the number of num_heads
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])
    
    def get_config(self):
        config = super().get_config().copy()
        return config

    def call(self, inputs):
        """Computes the multihead attention for the given query, key, and value.
        
        Parameters:
        - query, key, value (tf.Tensor): Input matrices.
        - mask (tf.Tensor): Mask to apply on attention scores.
        
        Returns:
        - output (tf.Tensor): The multihead attention output.
        - attention (tf.Tensor): The attention weights from the last attention head.
        """
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # 1. Linear transformations using WQ, WK, WV. 
        # Note: In encoder-to-decoder attention, the length of the query might differ from the length of key and value.
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 2. Split the heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 3. Compute the scaled dot-product attention
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # 4. Concatenate the heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
  
        # 5. Final linear transformation with WO
        outputs = self.dense(concat_attention)

        return outputs