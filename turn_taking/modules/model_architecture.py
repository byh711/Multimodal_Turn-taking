import tensorflow as tf
from modules.multihead_attention import *
from modules.positional_encoding import *
from modules.crossmodal_transformer import *


def overall(text_size, audio_size, vision_size, game_size, text_dim, audio_dim, vision_dim, game_dim, kernel_size, num_layers, dff, d_model, num_heads, dropout, output_bias, name="overall"):
    """Constructs the complete model architecture, which encompasses embeddings, dimension reductions, 
    LSTM layers, crossmodal transformers, and the final output layers.
    
    Parameters:
    - text_size, audio_size, vision_size, game_size (int): Sizes of the inputs for text, audio, vision, and game respectively.
    - text_dim, audio_dim, vision_dim, game_dim (int): Dimensions of the inputs for text, audio, vision, and game respectively.
    - kernel_size (int): Kernel size for Conv1D layers.
    - num_layers (int): Number of crossmodal layers to stack.
    - dff (int): Dimension of the feed-forward network's inner layer.
    - d_model (int): Dimension of the model's input and output.
    - num_heads (int): Number of attention heads.
    - dropout (float): Dropout rate.
    - output_bias (tf.keras.initializers.Initializer): Initializer for the bias of the output layer.
    - name (str, optional): Name of the model. Defaults to "overall".
    
    Returns:
    - tf.keras.Model: The complete model architecture.
    """
    # Embedding Vectors
    text_input = tf.keras.Input(shape=(text_size, text_dim), name="text_input")
    audio_input = tf.keras.Input(shape=(audio_size, audio_dim), name="audio_input") 
    vision_input = tf.keras.Input(shape=(vision_size, vision_dim),name="vision_input")
    game_input = tf.keras.Input(shape=(game_size, game_dim),name="game_input")
    
    
    # Dimension reduction
    text = tf.keras.layers.Conv1D(d_model, kernel_size, activation='linear', name="Conv1D_text")(text_input)
    audio = tf.keras.layers.Conv1D(d_model, kernel_size, activation='linear',name="Conv1D_audio")(audio_input)
    vision = tf.keras.layers.Conv1D(d_model, kernel_size, activation='linear',name="Conv1D_vision")(vision_input)
    game = tf.keras.layers.Conv1D(d_model, kernel_size, activation='linear',name="Conv1D_game")(game_input)

    
    # LSTM
    text = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(d_model, return_sequences=True, name="LSTM_text"))(text)
    audio = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(d_model, return_sequences=True, name="LSTM_audio"))(audio)
    vision = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(d_model, return_sequences=True, name="LSTM_vision"))(vision)
    game = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(d_model, return_sequences=True, name="LSTM_game"))(game)
    
    
    # Crossmodal Transformer
    audio_text = crossmodal(source_size=2*audio_size, target_size=2*text_size, num_layers=num_layers, dff=dff,
      d_model=2*d_model, num_heads=num_heads, dropout=dropout,name="audio_text")(inputs=[audio, text])        
    
    vision_text = crossmodal(source_size=2*vision_size, target_size=2*text_size, num_layers=num_layers, dff=dff,
      d_model=2*d_model, num_heads=num_heads, dropout=dropout,name="vision_text")(inputs=[vision, text])         
    
    game_text = crossmodal(source_size=2*game_size, target_size=2*text_size, num_layers=num_layers, dff=dff,
      d_model=2*d_model, num_heads=num_heads, dropout=dropout,name="game_text")(inputs=[game, text])      
    
    text_audio = crossmodal(source_size=2*text_size, target_size=2*audio_size, num_layers=num_layers, dff=dff,
      d_model=2*d_model, num_heads=num_heads, dropout=dropout,name="text_audio")(inputs=[text, audio])       
    
    vision_audio = crossmodal(source_size=2*vision_size, target_size=2*audio_size, num_layers=num_layers, dff=dff,
      d_model=2*d_model, num_heads=num_heads, dropout=dropout,name="vision_audio")(inputs=[vision, audio])    
    
    game_audio = crossmodal(source_size=2*game_size, target_size=2*audio_size, num_layers=num_layers, dff=dff,
      d_model=2*d_model, num_heads=num_heads, dropout=dropout,name="game_audio")(inputs=[game, audio])    
    
    text_vision = crossmodal(source_size=2*text_size, target_size=2*vision_size, num_layers=num_layers, dff=dff,
      d_model=2*d_model, num_heads=num_heads, dropout=dropout,name="text_vision")(inputs=[text, vision])      
    
    audio_vision = crossmodal(source_size=2*audio_size, target_size=2*vision_size, num_layers=num_layers, dff=dff,
      d_model=2*d_model, num_heads=num_heads, dropout=dropout,name="audio_vision")(inputs=[audio, vision])    
    
    game_vision = crossmodal(source_size=2*game_size, target_size=2*vision_size, num_layers=num_layers, dff=dff,
      d_model=2*d_model, num_heads=num_heads, dropout=dropout,name="game_vision")(inputs=[game, vision])    
    
    text_game = crossmodal(source_size=2*text_size, target_size=2*game_size, num_layers=num_layers, dff=dff,
      d_model=2*d_model, num_heads=num_heads, dropout=dropout,name="text_game")(inputs=[text, game])      
    
    audio_game = crossmodal(source_size=2*audio_size, target_size=2*game_size, num_layers=num_layers, dff=dff,
      d_model=2*d_model, num_heads=num_heads, dropout=dropout,name="audio_game")(inputs=[audio, game])    
    
    vision_game = crossmodal(source_size=2*vision_size, target_size=2*game_size, num_layers=num_layers, dff=dff,
      d_model=2*d_model, num_heads=num_heads, dropout=dropout,name="vision_game")(inputs=[vision, game])    
    
    
    # Fusion1
    text_fusion = tf.keras.layers.Concatenate(axis=2,name='text_fusion')([audio_text, vision_text, game_text])
    audio_fusion = tf.keras.layers.Concatenate(axis=2,name='audio_fusion')([text_audio, vision_audio, game_audio])
    vision_fusion = tf.keras.layers.Concatenate(axis=2,name='vision_fusion')([text_vision, audio_vision, game_vision])
    game_fusion = tf.keras.layers.Concatenate(axis=2,name='game_fusion')([text_game, audio_game, vision_game])
    
    
    # LSTM2
    text = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(d_model, name="BiLSTM_text"))(text_fusion)
    audio = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(d_model, name="BiLSTM_audio"))(audio_fusion)
    vision = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(d_model, name="BiLSTM_vision"))(vision_fusion)
    game = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(d_model, name="BiLSTM_game"))(game_fusion)

    
    # Fusion2
    fusion = tf.keras.layers.Concatenate(axis=1, name='fusion')([text, audio, vision, game])


    # Output Layer
    outputs = tf.keras.layers.Dense(64, name='outputs_dense1', activation='relu')(fusion)
    outputs = tf.keras.layers.Dropout(dropout)(outputs)
    outputs = tf.keras.layers.Dense(8, name='outputs_dense2')(outputs)
    outputs = tf.keras.layers.Dense(1, name="outputs",activation='sigmoid', bias_initializer=output_bias)(outputs)
    

    return tf.keras.Model(inputs=[text_input, audio_input, vision_input, game_input], outputs=[outputs], name=name)