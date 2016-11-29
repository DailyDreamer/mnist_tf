import math
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
# MNIST dataset paramaters
NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
TIME=datetime.now().strftime("%Y%m%d-%H%M%S")
# Basic model parameters as external flags.
flags = tf.app.flags
flags.DEFINE_string('train_dir', './mnist', 'Directory to put the training data.')
flags.DEFINE_string('summary_dir', './summary/mlp_relu/'+TIME, 'Directory to put the summary data.')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 30, 'Number of epochs to run trainer.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
FLAGS = flags.FLAGS


def weight_variable(shape, stddev):
  initial = tf.truncated_normal(shape=shape, stddev=stddev, name='weights')
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, name='biases')
  return tf.Variable(initial)

def main(_):
  # get mnist data
  # images will be normalize to [0.0, 1.0], with shape [EPOCH_SIZE, IMAGE_PIXELS]
  # labels will be one hot format, with shape [EPOCH_SIZE, NUM_CLASSES]
  mnist_data = read_data_sets(FLAGS.train_dir, one_hot=True, validation_size=0)

  images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS])
  labels_placeholder = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

  # build the graph
  with tf.name_scope('hidden1'):
    weights = weight_variable([IMAGE_PIXELS, FLAGS.hidden1], 
                              stddev=1.0/math.sqrt(float(IMAGE_PIXELS)))
    biases = bias_variable([FLAGS.hidden1])
    hidden1 = tf.nn.relu(tf.matmul(images_placeholder, weights) + biases)

  with tf.name_scope('hidden2'):
    weights = weight_variable([FLAGS.hidden1, FLAGS.hidden2], 
                              stddev=1.0/math.sqrt(float(FLAGS.hidden1)))    
    biases = bias_variable([FLAGS.hidden2])
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

  with tf.name_scope('softmax'):
    weights = weight_variable([FLAGS.hidden2, NUM_CLASSES], 
                              stddev=1.0/math.sqrt(float(FLAGS.hidden2)))    
    biases = bias_variable([NUM_CLASSES])
    logits_op = tf.matmul(hidden2, weights) + biases
    # tf.nn.softmax_cross_entropy_with_logits will compute softmax internal
    # but we need to add softmax when evaluation

  with tf.name_scope('loss'):  
    diff = tf.nn.softmax_cross_entropy_with_logits(logits_op, labels_placeholder)
    loss_op = tf.reduce_mean(diff, name='cross_entropy')
    tf.scalar_summary(loss_op.op.name, loss_op)    

  with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss_op, global_step=global_step)

  with tf.name_scope('evaluation'):
    logits_softmax = tf.nn.softmax(logits_op)
    num_correct = tf.equal(tf.argmax(logits_softmax, 1), tf.argmax(labels_placeholder, 1))
    accuracy_op = tf.reduce_mean(tf.cast(num_correct, tf.float32))
    tf.scalar_summary(accuracy_op.op.name, accuracy_op)        

  init_op = tf.initialize_all_variables()
  summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir)

  with tf.Session() as sess:
    sess.run(init_op)
    # start training
    num_batches = mnist_data.train.num_examples // FLAGS.batch_size
    for i in range(FLAGS.num_epochs * num_batches):
      x, y = mnist_data.train.next_batch(FLAGS.batch_size)
      train_feed = { images_placeholder: x, labels_placeholder: y }
      summary, _ = sess.run([summary_op, train_op], feed_dict=train_feed)
      summary_writer.add_summary(summary, i)      
      if i % num_batches == 0:
        test_feed = {
          images_placeholder: mnist_data.test.images, 
          labels_placeholder: mnist_data.test.labels
        }
        summary, accuracy = sess.run([summary_op, accuracy_op], feed_dict=test_feed)
        print('Accuracy at epoch %s: %s' % (i // num_batches, accuracy))

  summary_writer.close()
  
if __name__ == '__main__':
  tf.app.run()