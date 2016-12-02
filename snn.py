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
flags.DEFINE_string('summary_dir', './summary/snn/'+TIME, 'Directory to put the summary data.')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 30, 'Number of epochs to run trainer.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_float('V_min', 0, 'Reset value of membrane potential.')
flags.DEFINE_float('V_th_hidden1', 0.5, 'Spike threshold in hidden layer 1.')
flags.DEFINE_float('V_th_hidden2', 0.5, 'Spike threshold in hidden layer 2.')
flags.DEFINE_float('V_th_softmax', 0.5, 'Spike threshold in softmax layer.')
FLAGS = flags.FLAGS

def weight_variable(shape, stddev):
  initial = tf.truncated_normal(shape=shape, stddev=stddev, name='weights')
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, name='biases')
  return tf.Variable(initial)

def activiate_if(V_in, V_pre, V_th, shape):
  """
  Activite function of integrate-and-fire model
  """
  V_now = V_pre + V_in
  cond = tf.greater_equal(V_now, tf.constant(V_th, tf.float32, shape))
  new_spike = tf.select(cond, tf.ones(shape), tf.zeros(shape))
  new_V_pre = tf.select(cond, tf.constant(FLAGS.V_min, tf.float32, shape), V_now)
  return (new_spike, new_V_pre)

def main(_):
  # get mnist data
  # images will be normalize to [0.0, 1.0], with shape [EPOCH_SIZE, IMAGE_PIXELS]
  # labels will be one hot format, with shape [EPOCH_SIZE, NUM_CLASSES]
  mnist_data = read_data_sets(FLAGS.train_dir, one_hot=True, validation_size=0)

  images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS])
  labels_placeholder = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

  # build the graph
  with tf.name_scope('hidden1'):
    W_hidden1 = weight_variable([IMAGE_PIXELS, FLAGS.hidden1], 
                              stddev=1.0/math.sqrt(float(IMAGE_PIXELS)))
    b_hidden1 = bias_variable([FLAGS.hidden1])
    hidden1 = tf.nn.relu(tf.matmul(images_placeholder, W_hidden1))

  with tf.name_scope('hidden2'):
    W_hidden2 = weight_variable([FLAGS.hidden1, FLAGS.hidden2], 
                              stddev=1.0/math.sqrt(float(FLAGS.hidden1)))    
    b_hidden2 = bias_variable([FLAGS.hidden2])
    hidden2 = tf.nn.relu(tf.matmul(hidden1, W_hidden2))

  with tf.name_scope('softmax'):
    W_softmax = weight_variable([FLAGS.hidden2, NUM_CLASSES], 
                              stddev=1.0/math.sqrt(float(FLAGS.hidden2)))    
    b_softmax = bias_variable([NUM_CLASSES])
    logits_op = tf.matmul(hidden2, W_softmax)
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

  with tf.name_scope('SNN_evaluation'):
    # init variables
    shape_input = [mnist_data.test.num_examples, IMAGE_PIXELS]
    shape_hidden1 = [mnist_data.test.num_examples, FLAGS.hidden1]
    V_pre_hidden1 = tf.Variable(tf.constant(FLAGS.V_min, tf.float32, shape_hidden1))
    shape_hidden2 = [mnist_data.test.num_examples, FLAGS.hidden2]
    V_pre_hidden2 = tf.Variable(tf.constant(FLAGS.V_min, tf.float32, shape_hidden2))
    shape_softmax = [mnist_data.test.num_examples, NUM_CLASSES]
    V_pre_softmax = tf.Variable(tf.constant(FLAGS.V_min, tf.float32, shape_softmax))
    spike_count = tf.Variable(tf.zeros(shape_softmax))

    #spike generation
    cond = tf.greater(images_placeholder, tf.random_uniform(shape_input))  
    spike_gen = tf.select(cond, tf.ones(shape_input), tf.zeros(shape_input))

    V_in_hidden1 = tf.nn.relu(tf.matmul(spike_gen, W_hidden1))
    spike_hidden1, tmp = activiate_if(V_in_hidden1, V_pre_hidden1, FLAGS.V_th_hidden1, shape_hidden1)
    V_pre_hidden1 = V_pre_hidden1.assign(tmp)
 
    V_in_hidden2 = tf.nn.relu(tf.matmul(spike_hidden1, W_hidden2))
    spike_hidden2, tmp = activiate_if(V_in_hidden2, V_pre_hidden2, FLAGS.V_th_hidden2, shape_hidden2)
    V_pre_hidden2 = V_pre_hidden2.assign(tmp)

    logits_op = tf.matmul(spike_hidden2, W_softmax)
    V_in_softmax = tf.nn.softmax(logits_op)
    spike_softmax, tmp = activiate_if(V_in_softmax, V_pre_softmax, FLAGS.V_th_softmax, shape_softmax)
    V_pre_softmax = V_pre_softmax.assign(tmp)
    # spike counter
    spike_count = spike_count.assign_add(spike_softmax)    

    # evaluate total simulate time accuracy
    num_correct_snn = tf.equal(tf.argmax(spike_count, 1), tf.argmax(labels_placeholder, 1))
    accuracy_op_snn = tf.reduce_mean(tf.cast(num_correct_snn, tf.float32))

  init_op = tf.initialize_all_variables()
  summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir)

  with tf.Session() as sess:
    sess.run(init_op)
    # start training
    test_feed = {
      images_placeholder: mnist_data.test.images, 
      labels_placeholder: mnist_data.test.labels
    }
    num_batches = mnist_data.train.num_examples // FLAGS.batch_size
    for i in range(FLAGS.num_epochs * num_batches):
      x, y = mnist_data.train.next_batch(FLAGS.batch_size)
      train_feed = { images_placeholder: x, labels_placeholder: y }
      summary, _ = sess.run([summary_op, train_op], feed_dict=train_feed)
      summary_writer.add_summary(summary, i)      
      if i % num_batches == 0:
        summary, accuracy = sess.run([summary_op, accuracy_op], feed_dict=test_feed)
        print('Accuracy at epoch %s: %s' % (i // num_batches, accuracy))
    
    # tune MLP to SNN to evaluate
    # set total simulate to 100 ms
    for _ in range(100):
      sess.run(spike_count, feed_dict=test_feed)
    accuracy = sess.run(accuracy_op_snn, feed_dict=test_feed)
    print('Accuracy at SNN : %s' % (accuracy)) 

  summary_writer.close()
  
if __name__ == '__main__':
  tf.app.run()