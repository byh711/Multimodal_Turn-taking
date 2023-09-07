import tensorflow as tf
from modules.multihead_attention import MultiHead_Attention
from modules.positional_encoding import PositionalEncoding


def crossmodal_layer(dff, d_model, num_heads, dropout, name="crossmodal_layer"):
    """Defines a single cross-modal layer with multihead attention, dropout, normalization, 
    and feed-forward operations.
    
    Parameters:
    - dff (int): Dimension of the feed-forward network's inner layer.
    - d_model (int): Dimension of the model's input and output.
    - num_heads (int): Number of attention heads.
    - dropout (float): Dropout rate.
    - name (str): Name of the layer.
    
    Returns:
    - tf.keras.Model: A cross-modal layer model.
    """       
    
    source_inputs = tf.keras.Input(shape=(None, d_model), name="source_inputs")
    target_inputs = tf.keras.Input(shape=(None, d_model), name="target_inputs")
    
    source_inputs_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(source_inputs)
    target_inputs_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(target_inputs)
    
    # Crossmodal attention
    attention = MultiHead_Attention(d_model, num_heads, name="attention")(inputs={
            'query': target_inputs_norm, 'key': source_inputs_norm, 'value': source_inputs_norm, 'mask': None
        })

    # Dropout + residual connection and layer normalization
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + target_inputs_norm)
    
    # Position-wise feed-forward network
    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # Dropout + residual connection
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = outputs + attention
    
    return tf.keras.Model(inputs=[source_inputs, target_inputs], outputs=[outputs], name=name)


def crossmodal(source_size, target_size, num_layers, dff, d_model, num_heads, dropout, name='crossmodal'):
    """Constructs the full cross-modal transformer by stacking multiple instances of the crossmodal_layer.
    
    Parameters:
    - source_size (int): Size of the source input.
    - target_size (int): Size of the target input.
    - num_layers (int): Number of crossmodal layers to stack.
    - dff (int): Dimension of the feed-forward network's inner layer.
    - d_model (int): Dimension of the model's input and output.
    - num_heads (int): Number of attention heads.
    - dropout (float): Dropout rate.
    - name (str): Name of the model.
    
    Returns:
    - tf.keras.Model: A cross-modal transformer model.
    """    
    
    source_inputs = tf.keras.Input(shape=(None, d_model), name='source_inputs')
    target_inputs = tf.keras.Input(shape=(None, d_model), name='target_inputs')
    
    # Source positional encoding + dropout    
    source_embeddings = source_inputs * tf.math.sqrt(tf.cast(d_model, tf.float32))
    source_embeddings = PositionalEncoding(source_size, d_model)(source_embeddings)
    source_outputs = tf.keras.layers.Dropout(rate=dropout)(source_embeddings)
    
    # Target positional encoding + dropout    
    target_embeddings = target_inputs * tf.math.sqrt(tf.cast(d_model, tf.float32))
    target_embeddings = PositionalEncoding(target_size, d_model)(target_embeddings)
    target_outputs = tf.keras.layers.Dropout(rate=dropout)(target_embeddings)
    
    # Stack crossmodal layers
    for i in range(num_layers):
        target_outputs = crossmodal_layer(dff=dff, d_model=d_model, num_heads=num_heads,
                                          dropout=dropout, name='crossmodal_layer_{}'.format(i))(inputs=[source_outputs, target_outputs])

    return tf.keras.Model(inputs=[source_inputs, target_inputs], outputs=[target_outputs], name=name)