import tensorflow as tf

hparams = tf.contrib.training.HParams(
    lr = 0.02,  # Initial learning rate.
    batch_size=100,
    embed_size=300,  # alias = E,
    max_length=1358,
    num_cluster=100
)