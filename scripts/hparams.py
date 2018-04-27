import tensorflow as tf

hparams = tf.contrib.training.HParams(
    lr = 0.001,  # Initial learning rate.
    batch_size=30,
    embed_size=300,  # alias = E,
    max_length=1358,
    num_cluster=100
)