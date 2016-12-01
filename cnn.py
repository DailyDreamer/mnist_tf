import math
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
# MNIST dataset paramaters
NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
COLOR_CHANNEL = 1
TIME=datetime.now().strftime("%Y%m%d-%H%M%S")
# Basic model parameters as external flags.
flags = tf.app.flags
flags.DEFINE_string('train_dir', './mnist', 'Directory to put the training data.')
flags.DEFINE_string('summary_dir', './summary/cnn/'+TIME, 'Directory to put the summary data.')
# 0.1 for GradientDescentOptimizer
# flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
# 0.001 for AdamOptimizer
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 50, 'Number of epochs to run trainer.')
flags.DEFINE_integer('hidden1', 12, 'Number of feature maps in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of feature maps in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
FLAGS = flags.FLAGS

def weight_variable(shape):
  initial = tf.truncated_normal(shape=shape, stddev=0.1, name='weights')
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, name='biases')
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main(_):
  # get mnist data
  # images will be normalize to [0.0, 1.0], with shape [EPOCH_SIZE, IMAGE_PIXELS]
  # labels will be one hot format, with shape [EPOCH_SIZE, NUM_CLASSES]
  mnist_data = read_data_sets(FLAGS.train_dir, one_hot=True, validation_size=0)

  images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS])
  labels_placeholder = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
  keep_prob = tf.placeholder(tf.float32)

  # build the graph
  with tf.name_scope('hidden1'):
    W_conv1 = weight_variable([5, 5, COLOR_CHANNEL, FLAGS.hidden1])
    b_conv1 = bias_variable([FLAGS.hidden1])

    x_image = tf.reshape(images_placeholder, [-1,IMAGE_SIZE,IMAGE_SIZE, COLOR_CHANNEL])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = pool_2x2(h_conv1)

  with tf.name_scope('dropout1'):  
    h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob)

  with tf.name_scope('hidden2'):
    W_conv2 = weight_variable([5, 5, FLAGS.hidden1, FLAGS.hidden2])
    b_conv2 = bias_variable([FLAGS.hidden2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2)
    h_pool2 = pool_2x2(h_conv2)

  with tf.name_scope('dropout2'):  
    h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob)

  with tf.name_scope('full_connected'):
    num_fc = IMAGE_SIZE//4 * IMAGE_SIZE//4 * FLAGS.hidden2
    W_fc1 = weight_variable([num_fc, NUM_CLASSES])
    b_fc1 = bias_variable([NUM_CLASSES])

    h_pool2_flat = tf.reshape(h_pool2_drop, [-1, num_fc])
    logits_op = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    # tf.nn.softmax_cross_entropy_with_logits will compute softmax internal
    # but we need to add softmax when evaluation

  with tf.name_scope('loss'):  
    diff = tf.nn.softmax_cross_entropy_with_logits(logits_op, labels_placeholder)
    loss_op = tf.reduce_mean(diff, name='cross_entropy')
    tf.scalar_summary(loss_op.op.name, loss_op)    

  with tf.name_scope('train'):
    #optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
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
      train_feed = { images_placeholder: x, labels_placeholder: y, keep_prob: 0.5 }
      summary, _ = sess.run([summary_op, train_op], feed_dict=train_feed)
      summary_writer.add_summary(summary, i)      
      if i % num_batches == 0:
        test_feed = {
          images_placeholder: mnist_data.test.images, 
          labels_placeholder: mnist_data.test.labels,
          keep_prob: 1.0
        }
        summary, accuracy = sess.run([summary_op, accuracy_op], feed_dict=test_feed)
        print('Accuracy at epoch %s: %s' % (i // num_batches, accuracy))

  summary_writer.close()
  
if __name__ == '__main__':
  tf.app.run()