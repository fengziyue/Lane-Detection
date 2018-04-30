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
import math
from six.moves import xrange  # pylint: disable=redefined-builtin

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.


# Global constants describing the CIFAR-10 data set.

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 15981
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 7000
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0000001       # Initial learning rate.

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
  with open('val.txt') as fid:  
    content = fid.read()  
    content = content.split('\n')    
    content = content[:-1]
    valuequeue = tf.train.string_input_producer(content,shuffle=True)  
    value = valuequeue.dequeue() 
    dir, label1= tf.decode_csv(records=value, record_defaults=[['string'], ['']], field_delim=" ")  
    label1 = tf.string_to_number(label1, tf.int32)
    result.label=label1
    print(dir)
  imagecontent = tf.read_file(dir)  
  image = tf.image.decode_jpeg(imagecontent, channels=3)
  image = tf.image.resize_images(image,[256,320])
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
        batch_size=1,
        num_threads=num_preprocess_threads,
        capacity=50000,
        min_after_dequeue=2800)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=1,
  shapes=([256,320,3],[]),
        num_threads=num_preprocess_threads,
        capacity=50000)
  # Display the training images in the visualizer.
  #tf.image_summary('images', images,max_images=64)
  print(images)
  return images, tf.reshape(label_batch, [batch_size])
def inputs():
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  batch_size=1
  filenames = './val.txt'
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  # Read examples from files
  read_input=tf.placeholder(tf.uint8)
  read_input = read_cifar10('val.txt')
  reshaped_image=tf.placeholder(tf.float32)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(reshaped_image, read_input.label,
                                         min_queue_examples, 1,
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
    reshape = tf.reshape(conv5, [1,-1])
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
    weights = tf.get_variable('weights', shape=[500, 300],
                                          initializer=tf.contrib.layers.xavier_initializer())
    biases = _variable_on_cpu('biases', [300], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases)
    tf.scalar_summary('local4' + '/weight', tf.reduce_sum(weights)/(500*300))
    tf.scalar_summary('local4' + '/biases', tf.reduce_sum(biases)/300)
    tf.histogram_summary('local4' + '/activations', local4)
    #_activation_summary(local4)

  
  with tf.variable_scope('local6') as scope:
    local6=tf.placeholder("float")
    weights=tf.placeholder("float")
    weights = tf.get_variable('weights', shape=[300, 200],
                                          initializer=tf.contrib.layers.xavier_initializer())
    biases = _variable_on_cpu('biases', [200], tf.constant_initializer(0.1))
    local6 = tf.matmul(local4, weights) + biases
    #local6 = tf.tanh(local6)
    tf.scalar_summary('local6' + '/weight', tf.reduce_sum(weights)/(300))
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
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  loss=cross_entropy_mean
  tf.histogram_summary('labels' + '/activations', labels)
  tf.histogram_summary('local6' + '/activations', logits)
  tf.scalar_summary('loss', loss)
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
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / 1
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  #lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
   #                               global_step,
    #                              decay_steps,
     #                             LEARNING_RATE_DECAY_FACTOR,
      #                            staircase=True)
  lr=0.0001
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
    images, labels = inputs()

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
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=False))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter('/home/fzyue/Desktop/caffeendtoend/1', sess.graph)

    for step in xrange(5000):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = 1
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
      if step % 1000 == 0 or (step + 1) == 1000000:
        checkpoint_path = os.path.join('/home/fzyue/Desktop/caffeendtoend/1', 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


'''def main(argv=None):  # pylint: disable=unused-argument
  #cifar10.maybe_download_and_extract()
  if tf.gfile.Exists('/home/fzyue/Desktop/caffeendtoend/1'):
    tf.gfile.DeleteRecursively('/home/fzyue/Desktop/caffeendtoend/1')
  tf.gfile.MakeDirs('/home/fzyue/Desktop/caffeendtoend/1')
  train()
main()'''
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""






FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'evalwrite',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '1',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5*50000,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 981,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


def eval_once(summary_writer, summary_op,top_k_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  saver=tf.train.Saver()
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / 1))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * 1
      step = 0
      count=0
      with open('result.txt', 'w') as test_txt:
        before_time=time.time()
        while step < num_iter and not coord.should_stop():
          
          logits,labels,precisions=sess.run(top_k_op)
          #predictions=tf.nn.in_top_k(logits, labels, 1)
          step+=1
          if(precisions==True):
            count+=1
          print (precisions)
          '''line_logits='{} {} {} {} {}'.format(float(logits[0][0])*1280,float(logits[0][1])*1280,float(logits[0][2])*1280,float(logits[0][3])*1280,float(logits[0][4])*1024)
          line_dir = '%s'%(dir)
          line_dir=line_dir[10:-2]
          line=line_dir+' '+line_logits+'\n'
          test_txt.write(line)'''
          #print('one_next_time=%s ,run_time=%s ,all_time=%s ,'%(one_next_time,run_time,all_time,))
          #if step==499:
           # print('all_example_time=%s'%(float(time.time()-start_all_time)))
      # Compute precision @ 1.
      precision = 1
      print(count)
      print(ckpt)

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
def top_k(logits,labels):
  #print('labels=%s  logits=%s   error=%s'%(labels,logits,labels-logits))
  return logits,labels,tf.nn.in_top_k(logits, labels, 3)

def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    images=tf.placeholder("float",shape=[None,256,320,3])
    labels=tf.placeholder("float",shape=[None])
    images, labels = inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits=tf.placeholder("float",shape=[None])
    logits = inference(images)
    top_k_op=top_k(logits,labels)
    # Calculate predictions.
    '''top_k_op_logits = logits
    top_k_op_labels = labels[...,0]
    top_k_op_error = tf.sub(labels,logits)[...,0]'''
    #print('labels=%s  logits=%s error=%s'%(top_k_op_labels,top_k_op_logits,top_k_op_error))
    # Restore the moving average version of the learned variables for eval.
    '''variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)'''

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(summary_writer, summary_op,top_k_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


main()
