import tensorflow as tf

hparams = tf.contrib.training.HParams(
  lr = 0.001,  # Initial learning rate.
  batch_size=32,
  embed_size=300,  # alias = E,
  max_length=140
)