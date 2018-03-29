import tensorflow as tf
import os
import glob
import numpy as np
import codecs

from book2vecs.scripts.hparams import hparams

class B2P2VModel:
  def __init__(self):
    pass

  def model_fn(self, features, # This is batch_features from input_fn
                     labels,   # This is batch_labels from input_fn
                     mode,     # An instance of tf.estimator.ModeKeys
                     params):  # Additional configuration

    encoder_inputs = features['seq']
    encoder_inputs_length = features['seq_len']
    file_name = features['file_name']
    target_seq = features['target_seq']

    with tf.variable_scope("encoder"):
      cell = tf.contrib.rnn.GRUCell(hparams.embed_size)
      outputs, state_encoder = tf.nn.dynamic_rnn(cell=cell, inputs=encoder_inputs,
                                                 sequence_length=encoder_inputs_length, dtype=tf.float32)

    mask = tf.sequence_mask(encoder_inputs_length, dtype=tf.float32, maxlen=hparams.max_length)

    loss = tf.reduce_sum(tf.squared_difference(outputs, target_seq), axis=2) # [B, T]
    loss = loss * mask # [B, T]
    #loss = tf.Print(loss, [loss, tf.shape(loss)])

    total_loss = tf.reduce_mean(loss)

    tf.summary.scalar("state encoder norm", tf.norm(tf.reduce_mean(state_encoder, axis=0))) # mean over batches and norm of vector
    tf.summary.histogram("state encoder", tf.reduce_mean(state_encoder, axis=0))

    tf.summary.scalar("total_loss", total_loss)

    # total_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=target_seq, weights=mask)
    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(mode, loss=total_loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "book" : file_name,
            "state": state_encoder
        }
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, predictions=predictions)

    learning_rate = learning_rate_decay(init_lr=hparams.lr, global_step=tf.train.get_global_step())

    with tf.name_scope("train_op"):
      train_op = tf.contrib.training.create_train_op(total_loss=total_loss,
                                                     global_step=tf.train.get_global_step(),
                                                     transform_grads_fn=grads_clipping,
                                                     optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
                                                     summarize_gradients=True)

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


  def input_fn(self):

    def generator_lj():
        # load vectors
        par_vecs = np.load('../models/book_par_vecs_20k.npy')
        book_names = np.load('../models/book_filenames.npy')
        num_vec = np.load('../models/num_vec.npy')

        for name, par_vec, length in zip(book_names, par_vecs, num_vec):
            input_sequence = par_vec
            target_sequence = np.pad(par_vec[1:, :], [[0, 1], [0, 0]], mode='constant', constant_values=0)
            sequence_length = length
            file_name = name

            yield input_sequence, sequence_length, target_sequence, file_name

    dataset = tf.data.Dataset.from_generator(generator_lj,
                                             output_types=(tf.float32, tf.int32, tf.float32, tf.string),
                                             output_shapes=(tf.TensorShape([None, 300]),
                                                            tf.TensorShape([]),
                                                            tf.TensorShape([None, 300]),
                                                            tf.TensorShape([])))

    dataset = dataset.map(parse_example, num_parallel_calls=4)
    dataset = dataset.shuffle(buffer_size=256)
    #dataset = dataset.repeat(10)
    dataset = dataset.batch(hparams.batch_size)
    dataset = dataset.prefetch(5)

    return dataset

def parse_example(seq, seq_len, target_seq, file_name):
  return dict(
    seq=seq,
    seq_len=seq_len,
    target_seq=target_seq,
    file_name=file_name
  )

#def element_length_fn(features):
#  return features["encoder_input_len"]

def grads_clipping(grads_list):
  clipped = []
  for grad, var in grads_list:
    clipped.append((tf.clip_by_norm(grad, 5.), var))
  return clipped


def learning_rate_decay(init_lr, global_step, warmup_steps=4000.):
  '''Noam scheme from tensor2tensor'''
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


if __name__ == '__main__':
    b2p2vmodel = B2P2VModel()

    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    estimator_config = tf.estimator.RunConfig(session_config=session_config)

    classifier = tf.estimator.Estimator(
        model_fn=b2p2vmodel.model_fn,
        model_dir="../models/b2p2v_1",
        config=estimator_config,
        params={})

    train = False

    if train:
        epochs = 2000
        for ep in range(epochs):
            classifier.train(input_fn=b2p2vmodel.input_fn)
    else:
        predictions = classifier.predict(input_fn=b2p2vmodel.input_fn)
        book_filenames = []
        book_vecs = []
        for p in predictions:
            book_filenames.append(p['book'].decode("utf-8"))
            book_vecs.append(p['state'])

        book_vecs = Z = [x for _,x in sorted(zip(book_filenames,book_vecs))]
        book_filenames = sorted(book_filenames)

        np.save('../models/b2p2v_book_filenames.npy', book_filenames)
        np.save('../models/b2p2v_book_vecs.npy', book_vecs)