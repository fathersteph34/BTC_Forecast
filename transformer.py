import tensorflow as tf
import numpy as np

"""
ref : https://arminnorouzi.github.io/posts/2023/05/blog-post-13/
"""

def positional_encoding(length, depth):
  """
  Generates a matrix of position encodings for an input sequence.

  Args:
      length: An integer representing the length of the input sequence.
      depth: An integer representing the dimensionality of the encoding.

  Returns:
      A `tf.Tensor` of shape `(length, depth)` representing the position encoding matrix.
  """
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

  return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
  """
  This layer combines the input embedding with a positional encoding that helps the Transformer to understand
  the relative position of the tokens in a sequence. It takes an input sequence of tokens and converts it to
  a sequence of embedding vectors, then adds positional information to it.

  Attributes:
      vocab_size (int): The size of the vocabulary, i.e., the number of unique tokens in the input sequence.
      d_model (int): The number of dimensions in the embedding vector.

  Methods:
      compute_mask(*args, **kwargs): This method computes the mask to be applied to the embeddings.
      call(x): This method performs the computation for the layer.

  """
  def __init__(self, vocab_size, d_model):
    """
    Initializes the PositionalEmbedding layer.

    Args:
        vocab_size (int): The size of the vocabulary, i.e., the number of unique tokens in the input sequence.
        d_model (int): The number of dimensions in the embedding vector.
    """
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    """
    Computes the mask to be applied to the embeddings.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Mask to be applied to the embeddings.
    """
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    """
    Computes the output of the layer.

    Args:
        x (tf.Tensor): Input sequence of tokens.

    Returns:
        The output sequence of embedding vectors with added positional information.
    """
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x
  

class BaseAttention(tf.keras.layers.Layer):
  """
  Base Attention layer class that contains a MultiHeadAttention, LayerNormalization and Add layer.

  Attributes:
  -----------
  kwargs: dict
      keyword arguments that will be passed to the MultiHeadAttention layer during initialization.

  Methods:
  --------
  call(inputs, mask=None, training=None):
      Performs a forward pass on the input and returns the output.

  """
  def __init__(self, **kwargs):
    """
    Initializes a new instance of the BaseAttention layer class.

    Parameters:
    -----------
    kwargs: dict
        keyword arguments that will be passed to the MultiHeadAttention layer during initialization.
    """
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
  """
  A class that implements cross-attention mechanism by inheriting from BaseAttention class.
  Cross-attention is used to process two different sequences and attends to the context sequence while processing the query sequence.
  Inherits:
      BaseAttention: A base class that defines the MultiHeadAttention layer, LayerNormalization, and Add operation.
  Args:
      **kwargs: Arguments to pass to the MultiHeadAttention layer.
  """
  def call(self, x, context):
    """
    The call function that performs the cross-attention operation.

    Args:
        x: The query sequence tensor, shape=(batch_size, seq_len, embedding_dim)
        context: The context sequence tensor, shape=(batch_size, seq_len, embedding_dim)

    Returns:
        The attended output tensor, shape=(batch_size, seq_len, embedding_dim)
    """
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
  
class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    """
    Apply the global self-attention mechanism to the input sequence.

    Args:
        x: A tensor of shape `(batch_size, seq_len, embedding_dim)`
        representing the input sequence.

    Returns:
        A tensor of the same shape as the input, representing the sequence
        after being transformed by the self-attention mechanism.
    """
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  
class CausalSelfAttention(BaseAttention):
  """
  Call self attention on the input sequence, ensuring that each position in the
  output depends only on previous positions (i.e. a causal model).

  Args:
      x: Input sequence tensor of shape `(batch_size, seq_len, embed_dim)`.

  Returns:
      Output sequence tensor of the same shape as the input, after self-attention
      and residual connection with layer normalization applied.
  """
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  

class FeedForward(tf.keras.layers.Layer):
  """
  Implements the feedforward sublayer of the transformer block.

  Parameters:
  -----------
  d_model: int
      The number of expected features in the input and output.
  dff: int
      The number of neurons in the first Dense layer.
  dropout_rate: float, optional (default=0.1)
      The dropout rate to use.

  Attributes:
  -----------
  seq: tf.keras.Sequential
      The sequential model that applies the two Dense layers and Dropout.
  add: tf.keras.layers.Add
      The addition layer that adds the residual connection.
  layer_norm: tf.keras.layers.LayerNormalization
      The normalization layer applied to the output.

  Methods:
  --------
  call(x):
      Computes the feedforward sublayer on the input tensor x and returns the output.

  """
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    """
    Passes the input tensor `x` through a feedforward network consisting of two
    dense layers with `dff` hidden units and a `relu` activation function.
    A `dropout_rate` is applied after the first dense layer to prevent overfitting.
    The output of the feedforward network is added to the original input `x` via the
    `Add()` layer. Finally, the output is normalized using the `LayerNormalization()` layer.

    Args:
        x (tf.Tensor): Input tensor with shape `(batch_size, seq_len, d_model)`.

    Returns:
        tf.Tensor: Output tensor with shape `(batch_size, seq_len, d_model)`.
    """
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x)
    return x


class EncoderLayer(tf.keras.layers.Layer):
  """
  A single layer in the transformer encoder stack.

  Args:
    d_model (int): The dimensionality of the input and output sequences.
    num_heads (int): The number of attention heads to be used in the self-attention sub-layer.
    dff (int): The number of hidden units in the feedforward sub-layer.
    dropout_rate (float): The dropout rate to be applied after the self-attention sub-layer.

  Attributes:
    self_attention (GlobalSelfAttention): A self-attention layer.
    ffn (FeedForward): A feedforward neural network layer.
  """
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    """
    Applies the forward pass of the encoder layer.

    Args:
      x (tf.Tensor): The input sequence tensor.

    Returns:
      tf.Tensor: The output sequence tensor.
    """
    x = self.self_attention(x)
    x = self.ffn(x)
    return x


class Encoder(tf.keras.layers.Layer):
  """
  A custom Keras layer that implements the encoder of a transformer-based
  neural network architecture for natural language processing tasks such
  as language translation or text classification.

  Args:
    num_layers (int): The number of layers in the encoder.
    d_model (int): The dimensionality of the output space.
    num_heads (int): The number of attention heads in the multi-head
      self-attention mechanism.
    dff (int): The dimensionality of the fully connected feedforward
      network.
    vocab_size (int): The size of the vocabulary of the input language.
    dropout_rate (float): The dropout rate to use for regularization.

  Attributes:
    d_model (int): The dimensionality of the output space.
    num_layers (int): The number of layers in the encoder.
    pos_embedding (PositionalEmbedding): The layer that learns the position
      embeddings for each token in the input sequence.
    enc_layers (list): A list of `EncoderLayer` instances, one for each
      layer in the encoder architecture.
    dropout (Dropout): The dropout layer for regularization.

  Methods:
    call(x): The forward pass of the encoder layer.

  Returns:
    The output tensor of the encoder layer, which has shape
    `(batch_size, seq_len, d_model)`.
  """
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    

  def call(self, x):
    """
    Perform forward pass of the `Encoder` layer.

    Args:
    x: tensor of shape (batch_size, sequence_length) representing the input token IDs sequence.

    Returns:
    A tensor of shape (batch_size, sequence_length, d_model) representing the output after applying
    the self-attention and feed-forward layers to the input sequence.
    """

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)