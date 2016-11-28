import os.path
import time

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import mnist


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', './mnist', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit test')


def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
  mnist_data = read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

    logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
    loss = mnist.loss(logits, labels_placeholder)
    train_op = mnist.training(loss, FLAGS.learning_rate)
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    summary = tf.merge_all_summaries()
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in range(FLAGS.max_steps):
      start_time = time.time()
      feed_dict = fill_feed_dict(mnist_data.train, images_placeholder, labels_placeholder)
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
      duration = time.time() - start_time
      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=step)
        print('Training Data Eval:')
        do_eval(sess, eval_correct, images_placeholder, labels_placeholder, mnist_data.train)
        print('Validation Data Eval:')
        do_eval(sess, eval_correct, images_placeholder, labels_placeholder, mnist_data.validation)
        print('Test Data Eval:')
        do_eval(sess, eval_correct, images_placeholder, labels_placeholder, mnist_data.test)

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
