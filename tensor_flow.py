"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import re
import sys
import tarfile
import numpy as np
from six.moves import urllib
import tensorflow as tf
import os
from datetime import datetime
import os.path
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.


# Global constants describing the CIFAR-10 data set.

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 15000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """
  

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  result.height = 256
  result.width = 320
  result.depth = 3
  with open('train.txt') as fid:  
    content = fid.read()  
    content = content.split('\n')    
    content = content[:-1]
    valuequeue = tf.train.string_input_producer(content,shuffle=True)  
    value = valuequeue.dequeue() 
    dir, label1,label2,label3,label4,label5,label6= tf.decode_csv(records=value, record_defaults=[['string'], [''],[''],[''],[''],[''],['']], field_delim=" ")  
    label1 = tf.string_to_number(label1, tf.float32)
    label2 = tf.string_to_number(label2, tf.float32)
    label3 = tf.string_to_number(label3, tf.float32)
    label4 = tf.string_to_number(label4, tf.float32)
    label5 = tf.string_to_number(label5, tf.float32)
    label6 = tf.string_to_number(label6, tf.float32)
    result.label=tf.pack([label1,label2,label3,label4,label5])
    print(dir)
  imagecontent = tf.read_file(dir)  
  image = tf.image.decode_jpeg(imagecontent, channels=3)  
  result.uint8image=image
  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  print('generate')
  num_preprocess_threads = 8
  images=tf.placeholder(tf.float32)
  label_batch=tf.placeholder(tf.float32)
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=64,
        num_threads=num_preprocess_threads,
        capacity=50000,
        min_after_dequeue=2800)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=64,
  shapes=([256,320,3],[5]),
        num_threads=num_preprocess_threads,
        capacity=50000)
  # Display the training images in the visualizer.
  #tf.image_summary('images', images,max_images=64)
  print(images)
  return images, tf.reshape(label_batch, [batch_size,5])
def inputs():
  print('input')
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  batch_size=64
  filenames = './train.txt'
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  # Read examples from files
  read_input=tf.placeholder(tf.uint8)
  read_input = read_cifar10('train.txt')
  reshaped_image=tf.placeholder(tf.float32)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(reshaped_image, read_input.label,
                                         min_queue_examples, 64,
                                         shuffle=False)
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/gpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var
def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    norm1=tf.placeholder("float",shape=[None,256,320,3])
    conv1=tf.placeholder("float")
    conv=tf.placeholder("float")
    bias=tf.placeholder("float")
    #norm1=images
    norm1 = tf.nn.lrn(images, 4, bias=255.0, alpha=0.0, beta=1.0,
                    name='norm1')
    norm1=norm1-0.5
    tf.histogram_summary('norm1' + '/activations', norm1)
    kernel = tf.get_variable('weights',
                                         shape=[5, 5, 3, 24],
                                         initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(norm1, kernel, [1, 2, 2, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [24], tf.constant_initializer(0.1))
    weight=tf.reduce_sum(kernel)/(5*5*3*24)
    biases_ave=tf.reduce_sum(biases)/24
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias)
    tf.scalar_summary('conv1' + '/weight', weight)
    tf.scalar_summary('conv1' + '/biases', biases_ave)
    tf.histogram_summary('conv1' + '/activations', conv1)
    tf.image_summary('conv1', images,max_images=24)
    #tf.image_summary('conv1', tf.transpose(conv1, [3, 1, 2, 0])[...,0:1],max_images=24)
    #_activation_summary(conv1)
  # conv2
  with tf.variable_scope('conv2') as scope:
    conv2=tf.placeholder("float")
    conv=tf.placeholder("float")
    bias=tf.placeholder("float")
    kernel = tf.get_variable('weights',
                                         shape=[5, 5, 24, 36],
                                         initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(conv1, kernel, [1, 2, 2, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [36], tf.constant_initializer(0.1))
    weight=tf.reduce_sum(kernel)/(5*5*36*24)
    biases_ave=tf.reduce_sum(biases)/36
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias)
    tf.scalar_summary('conv2' + '/weight', weight)
    tf.scalar_summary('conv2' + '/biases', biases_ave)
    tf.histogram_summary('conv2' + '/activations', conv2)
    tf.image_summary('conv2', tf.transpose(conv2, [3, 1, 2, 0])[...,0:1],max_images=36)
    #_activation_summary(conv2)
  # conv3
  with tf.variable_scope('conv3') as scope:
    conv3=tf.placeholder("float")
    conv=tf.placeholder("float")
    bias=tf.placeholder("float")
    kernel = tf.get_variable('weights',
                                         shape=[5, 5, 36, 48],
                                         initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [48], tf.constant_initializer(0.1))
    weight=tf.reduce_sum(kernel)/(5*5*36*48)
    biases_ave=tf.reduce_sum(biases)/48
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias)
    tf.scalar_summary('conv3' + '/weight', weight)
    tf.scalar_summary('conv3' + '/biases', biases_ave)
    tf.histogram_summary('conv3' + '/activations', conv3)
    tf.image_summary('conv3', tf.transpose(conv3, [3, 1, 2, 0])[...,0:1],max_images=48)
    #_activation_summary(conv3)
  # conv4
  with tf.variable_scope('conv4') as scope:
    conv4=tf.placeholder("float")
    conv=tf.placeholder("float")
    bias=tf.placeholder("float")
    kernel = tf.get_variable('weights',
                                         shape=[3, 3, 48, 64],
                                         initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    weight=tf.reduce_sum(kernel)/(3*3*48*64)
    biases_ave=tf.reduce_sum(biases)/64
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias)
    tf.scalar_summary('conv4' + '/weight', weight)
    tf.scalar_summary('conv4' + '/biases', biases_ave)
    tf.histogram_summary('conv4' + '/activations', conv4)
    tf.image_summary('conv4', tf.transpose(conv4, [3, 1, 2, 0])[...,0:1],max_images=64)
    #_activation_summary(conv4)
  # conv5
  with tf.variable_scope('conv5') as scope:
    conv5=tf.placeholder("float")
    conv=tf.placeholder("float")
    bias=tf.placeholder("float")
    kernel = tf.get_variable('weights',
                                         shape=[3, 3, 64, 128],
                                         initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    weight=tf.reduce_sum(kernel)/(3*3*64*64)
    biases_ave=tf.reduce_sum(biases)/128
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias)
    tf.scalar_summary('conv5' + '/weight', weight)
    tf.scalar_summary('conv5' + '/biases', biases_ave)
    tf.histogram_summary('conv5' + '/activations', conv5)
    tf.image_summary('conv5', tf.transpose(conv5, [3, 1, 2, 0])[...,0:1],max_images=64)
  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    local3=tf.placeholder("float")
    dim=tf.placeholder(tf.int32)
    bias=tf.placeholder("float")
    weights=tf.placeholder("float")
    reshape = tf.reshape(conv5, [64,-1])
    dim = reshape.get_shape()[1].value
    weights = tf.get_variable('weights', shape=[dim, 500],
                                          initializer=tf.contrib.layers.xavier_initializer())
    biases = _variable_on_cpu('biases', [500], tf.constant_initializer(0.1))
    #local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases,name=scope.name)
    bias = tf.matmul(reshape, weights)+biases
    local3=tf.nn.relu(bias)
    tf.scalar_summary('local3' + '/weight', tf.reduce_sum(weights)/(dim*100))
    tf.scalar_summary('local3' + '/biases', tf.reduce_sum(biases)/100)
    tf.histogram_summary('local3' + '/activations', local3)
    #_activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    local4=tf.placeholder("float")
    weights=tf.placeholder("float")
    weights = tf.get_variable('weights', shape=[500, 100],
                                          initializer=tf.contrib.layers.xavier_initializer())
    biases = _variable_on_cpu('biases', [100], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases)
    tf.scalar_summary('local4' + '/weight', tf.reduce_sum(weights)/(500*100))
    tf.scalar_summary('local4' + '/biases', tf.reduce_sum(biases)/100)
    tf.histogram_summary('local4' + '/activations', local4)
    #_activation_summary(local4)
  #local5
  with tf.variable_scope('local5') as scope:
    local5=tf.placeholder("float")
    weights=tf.placeholder("float")
    weights = tf.get_variable('weights', shape=[100, 20],
                                          initializer=tf.contrib.layers.xavier_initializer())
    biases = _variable_on_cpu('biases', [20], tf.constant_initializer(0.1))
    local5 = tf.nn.relu(tf.matmul(local4, weights) + biases)
    tf.scalar_summary('local5' + '/weight', tf.reduce_sum(weights)/(20*100))
    tf.scalar_summary('local5' + '/biases', tf.reduce_sum(biases)/20)
    tf.histogram_summary('local5' + '/activations', local5)
    #_activation_summary(local5)
  with tf.variable_scope('local6') as scope:
    local6=tf.placeholder("float")
    weights=tf.placeholder("float")
    weights = tf.get_variable('weights', shape=[20, 5],
                                          initializer=tf.contrib.layers.xavier_initializer())
    biases = _variable_on_cpu('biases', [5], tf.constant_initializer(0.1))
    local6 = tf.matmul(local5, weights) + biases
    #local6 = tf.tanh(local6)
    tf.scalar_summary('local6' + '/weight', tf.reduce_sum(weights)/(20))
    tf.scalar_summary('local6' + '/biases', tf.reduce_sum(biases))
    # tf.histogram_summary('local6' + '/activations', local6)
    #_activation_summary(local6)
    #local6=local6[...,0]
  return local6
def losss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:queue
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  loss=tf.placeholder("float")
  loss = tf.reduce_sum(tf.pow(tf.sub(labels,logits), 2))/128
  # loss= tf.reduce_sum(tf.pow(labels-logits,2))
  # loss=tf.nn.l2_loss(labels,logits)
  tf.histogram_summary('labels' + '/activations', labels)
  tf.histogram_summary('local6' + '/activations', logits)
  tf.scalar_summary('loss', loss)
  tf.histogram_summary('local6-labels' + '/activations', tf.sub(logits,labels))
  return loss

def trainn(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / 64
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  #lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
   #                               global_step,
    #                              decay_steps,
     #                             LEARNING_RATE_DECAY_FACTOR,
      #                            staircase=True)
  lr=0.001
  tf.scalar_summary('learning_rate', lr)
  # Generate moving averages of all losses and associated summaries.
  #loss_averages_op = tf.reduce_sum(total_loss)

  # Compute gradients.
  #with tf.control_dependencies([loss_averages_op]):
  opt = tf.train.GradientDescentOptimizer(lr)
  grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  #tf.scalar_summary('grad', grads)
  #tf.histogram_summary('grads' + '/activations', grads)
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  '''# Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')'''
  train_op=apply_gradient_op
  return train_op
def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images=tf.placeholder("float",shape=[None,256,320,3])
    labels=tf.placeholder("float",shape=[None])
    local6=tf.placeholder("float",shape=[None])
    images, labels = inputs()
    print('images')
    print(images)
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = inference(images)

    # Calculate loss.
    loss = losss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = trainn(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    tf_config=tf.ConfigProto(
        log_device_placement=False)
    tf_config.gpu_options.per_process_gpu_memory_fraction=0.9
    sess = tf.Session(config=tf_config)
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter('/home/fzyue/Desktop/caffeendtoend/1', sess.graph)

    for step in xrange(100000):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = 64
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))
        #print(labels)
        #print (sess.run(logits))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == 100000:
        checkpoint_path = os.path.join('/home/fzyue/Desktop/caffeendtoend/1', 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  #cifar10.maybe_download_and_extract()
  if tf.gfile.Exists('/home/fzyue/Desktop/caffeendtoend/1'):
    tf.gfile.DeleteRecursively('/home/fzyue/Desktop/caffeendtoend/1')
  tf.gfile.MakeDirs('/home/fzyue/Desktop/caffeendtoend/1')
  train()
main()
