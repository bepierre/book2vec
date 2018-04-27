import tensorflow as tf
import os
import glob
import numpy as np
import codecs
import scipy.io

from book2vecs.scripts.hparams import hparams

class BPCModel:
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

        emb_encoder_inputs = tf.one_hot(encoder_inputs, depth=hparams.num_cluster, axis=-1)

        with tf.variable_scope("encoder"):
            cell = tf.contrib.rnn.GRUCell(hparams.num_cluster)
            outputs, state_encoder = tf.nn.dynamic_rnn(cell=cell, inputs=emb_encoder_inputs,
                                                       sequence_length=encoder_inputs_length, dtype=tf.float32)

        mask = tf.sequence_mask(encoder_inputs_length, dtype=tf.float32, maxlen=hparams.max_length)

        loss = mask * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=target_seq)) # [B, T]
        #loss = loss * mask # [B, T]
        #loss = tf.Print(loss, [loss, tf.shape(loss)])

        total_loss = tf.reduce_mean(loss)

        tf.summary.scalar("state encoder norm", tf.norm(tf.reduce_mean(state_encoder, axis=0))) # mean over batches and norm of vector
        # tf.summary.histogram("state encoder", tf.reduce_mean(state_encoder, axis=0))

        tf.summary.scalar("total_loss", total_loss)

        class_probabilities = tf.nn.softmax(outputs)

        acc = tf.metrics.accuracy(labels=target_seq,
                                  predictions=tf.argmax(class_probabilities, axis=2, output_type=tf.int32),
                                  name="accuracy",
                                  weights=mask)
        #tf.summary.scalar("accuracy", acc[0])
        metrics = {"accuracy": acc}

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)



        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'book': file_name,
                'state': state_encoder,
                'probs': class_probabilities,
                'target':target_seq,
                'size':encoder_inputs_length
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


    def input_fn(self, mode):

        def generator_lj(mode):
            # load vectors
            if(mode=='train'):
                labels = np.load('../models/book_labels_full_4c_w.npy')
                book_names = np.load('../models/book_filenames_full_4c_w.npy')
                num_vec = np.load('../models/num_vec_full_4c_w.npy')
            elif(mode=='eval'):
                labels = np.load('../models/eval_labels_full_4c_w.npy')
                book_names = np.load('../models/eval_book_filenames_full_4c_w.npy')
                num_vec = np.load('../models/eval_num_vec_full_4c_w.npy')
            elif(mode=='predict'):
                labels = np.load('../models/eval_labels_full_4c_w.npy')
                book_names = np.load('../models/eval_book_filenames_full_4c_w.npy')
                num_vec = np.load('../models/eval_num_vec_full_4c_w.npy')

            for name, labels, length in zip(book_names, labels, num_vec):
                input_sequence = labels
                target_sequence = np.pad(labels[1:], [[0, 1]], mode='constant', constant_values=0)
                sequence_length = length
                file_name = name

                yield input_sequence, sequence_length, target_sequence, file_name

        dataset = tf.data.Dataset.from_generator(lambda: generator_lj(mode),
                                                 output_types=(tf.int32, tf.int32, tf.int32, tf.string),
                                                 output_shapes=(tf.TensorShape([None]),
                                                                tf.TensorShape([]),
                                                                tf.TensorShape([None]),
                                                                tf.TensorShape([])))

        dataset = dataset.map(parse_example, num_parallel_calls=4)
        dataset = dataset.shuffle(buffer_size=128)
        # if mode=='train':
        #     dataset = dataset.repeat(5)
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
    bpcmodel = BPCModel()

    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    estimator_config = tf.estimator.RunConfig(session_config=session_config)

    classifier = tf.estimator.Estimator(
        model_fn=bpcmodel.model_fn,
        model_dir='../models/bpc_100',
        config=estimator_config,
        params={})

    train = True

    if train:
        epochs = 10000
        for ep in range(epochs):
            classifier.train(input_fn=lambda: bpcmodel.input_fn('train'))
            classifier.evaluate(input_fn=lambda: bpcmodel.input_fn('eval'))
    else:
        predictions = classifier.predict(input_fn=lambda: bpcmodel.input_fn('predict'))
        book_filenames = []
        label_probs = []
        target = []
        for p in predictions:
            size = p['size']
            book_filenames.append(p['book'].decode('utf-8'))
            label_probs.append(p['probs'][:size])
            target.append(p['target'][:size])

        label_probs = [x for _,x in sorted(zip(book_filenames,label_probs))]
        target = [x for _, x in sorted(zip(book_filenames, target))]
        book_filenames = sorted(book_filenames)

        books = np.zeros((len(book_filenames),), dtype=np.object)
        for i in range(len(book_filenames)):
            books[i] = {}
            books[i]['namse'] = book_filenames[i]
            books[i]['target'] = target[i]
            books[i]['probs'] = label_probs[i]

        scipy.io.savemat('../matlab/bpc/bpc_eval_books.mat', {'books': books})

        # list2 = np.array(book_filenames, dtype=np.object)
        # scipy.io.savemat('../matlab/kmeans/eval_book_names.mat', mdict={'eval_book_names': list2})
