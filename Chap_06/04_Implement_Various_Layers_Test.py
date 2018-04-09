import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

sess = tf.Session()

data_size = 25
stride_size = 1
conv_size = 5
maxpool_size = 5
conv_stride_size = 2
# Make 1 dimension array data
data_1d = np.random.normal(size=data_size)
x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])


# Define 1 dimension convolution layer
def conv_layer_1d(input_1d, my_filter, stride):
    # Expand 1 dimension array to vecter[# of data, width, height, # of channel]
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Process convolution
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1, 1, 1, 1], padding='VALID')

    conv_output_1d = tf.squeeze(convolution_output)
    return conv_output_1d


# Define filter [width, height, input, output]
my_filter = tf.Variable(tf.random_normal(shape=[1, conv_size, 1, 1]))
# Convolution Output for 1 dimension array data
my_convolution_output = conv_layer_1d(x_input_1d, my_filter, stride=1)


# Using relu activate function
def activation(input_1d):
    return tf.nn.relu(input_1d)


# Collecting func to activation func from conv_layer_1d
my_activation_output = activation(my_convolution_output)


# Define max_pool function for 1 dimension array data
def max_pool(input_1d, width, stride):
    # Expand 1 dimension array to vecter[# of data, width, height, # of channel]
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Process max_pooling
    pool_output = tf.nn.max_pool(input_4d, ksize=[1, 1, width, 1], strides=[1, 1, 1, 1], padding='VALID')
    # Reduce 4 dimension vecter data to 1 dimension array data
    pool_output_1d = tf.squeeze(pool_output)
    return pool_output_1d


# Collecting func to max_pool func from activation func
my_maxpool_output = max_pool(my_activation_output, width=maxpool_size)


# Define fully_connected layer with input_layer
def fully_connected(input_layer, num_outputs):
    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_outputs]]))
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_outputs])

    input_layer_2d = tf.expand_dims(input_layer, 0)

    full_output = tf.add(tf.matmul(input_layer_2d, weight), bias)

    full_output_1d = tf.squeeze(full_output)
    return full_output_1d


# Result
my_full_output = fully_connected(my_maxpool_output, 5)

# Init Variables
init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input_1d: data_1d}
# Print result
print('>>>> 1D Data <<<<')

print('Input = array of length %d' % (x_input_1d.shape.as_list()[0]))
print('Convolution w/ filter, length = %d, stride size = %d, results in an array of length %d:' % (
    conv_size, stride_size, my_convolution_output.shape.as_list()[0]))
print(sess.run(my_convolution_output, feed_dict=feed_dict))

print('\nInput = above array of length %d' % (my_convolution_output.shape.as_list()[0]))
print('ReLU element wise returns an array of length %d:' % (my_activation_output.shape.as_list()[0]))
print(sess.run(my_activation_output, feed_dict=feed_dict))

print('\nInput = above array of length %d' % (my_activation_output.shape.as_list()[0]))
print('MaxPool, window length = %d, stride size = %d, results in the array of length %d' % (
    maxpool_size, stride_size, my_maxpool_output.shape.as_list()[0]))
print(sess.run(my_maxpool_output, feed_dict=feed_dict))

print('\nInput = above array of length %d' % (my_maxpool_output.shape.as_list()[0]))
print('Fully connected layer on all 4 rows with %d outputs:' % (my_full_output.shape.as_list()[0]))
print(sess.run(my_full_output, feed_dict=feed_dict))

print('\n\n\n>>>> 2D Data <<<<')

ops.reset_default_graph()
sess = tf.Session()

row_size = 10
col_size = 10
maxpool_stride_size = 1
maxpool_size = 2

data_size = [row_size, col_size]
data_2d = np.random.normal(size=data_size)
x_input_2d = tf.placeholder(dtype=tf.float32, shape=data_size)


# Define 2 dimension convolution layer
def conv_layer_2d(input_2d, my_filter):
    # So start from input_2d but also expand dimension to vecter [# of data, width, height, # of channel]
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Different strides dimension [1, 2, 2, 1] prior things(1-dimension) [1, 1, 1, 1]
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1, 2, 2, 1],
                                      padding='VALID')
    conv_output_2d = tf.squeeze(convolution_output)
    return conv_output_2d


# Define filter [width, height, input, output]
my_filter = tf.Variable(tf.random_normal(shape=[2, 2, 1, 1]))
# Convolution layer for 2-dimension vecter
my_convolution_output = conv_layer_2d(x_input_2d, my_filter)

# Using relu func
def activation(input_1d):
    return tf.nn.relu(input_1d)

# Collecting func to activation func(relu) from conv_layer_2d
my_activation_output = activation(my_convolution_output)

# Define max_pooling func for 2-dimension vecter
def max_pool(input_2d, width, height):
    # Expand dimension
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    pool_output = tf.nn.max_pool(input_4d, ksize=[1, height, width, 1], strides=[1, 1, 1, 1], padding='VALID')
    pool_output_2d = tf.squeeze(pool_output)
    return pool_output_2d

# Collecting func to max_pool func from activation func(relu)
my_maxpool_output = max_pool(my_activation_output, width=maxpool_size, height=maxpool_size)

#Define fully_connected func
def fully_connected(input_layer, num_outputs):
    flat_input = tf.reshape(input_layer, [-1])

    weight_shape = tf.squeeze(tf.stack([tf.shape(flat_input), [num_outputs]]))
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_outputs])
    input_2d = tf.expand_dims(flat_input, 0)

    full_output = tf.add(tf.matmul(input_2d, weight), bias)

    full_output_2d = tf.squeeze(full_output)
    return full_output_2d

#Result
my_full_output = fully_connected(my_maxpool_output, 5)

#init Variables
init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input_2d: data_2d}
#Print Result
print('Input = %s array' % (x_input_2d.shape.as_list()))
print('%s Convolution, stride size = [%d, %d] , results in the %s array' % (
    my_filter.get_shape().as_list()[:2], conv_stride_size, conv_stride_size, my_convolution_output.shape.as_list()))
print(sess.run(my_convolution_output, feed_dict=feed_dict))

print('\nInput = the above %s array' % (my_convolution_output.shape.as_list()))
print('ReLU element wise returns the %s array' % (my_activation_output.shape.as_list()))
print(sess.run(my_activation_output, feed_dict=feed_dict))

print('\nInput = the above %s array' % (my_activation_output.shape.as_list()))
print('MaxPool, stride size = [%d, %d], results in %s array' % (
    maxpool_stride_size, maxpool_stride_size, my_maxpool_output.shape.as_list()))
print(sess.run(my_maxpool_output, feed_dict=feed_dict))

print('\nInput = the above %s array' % (my_maxpool_output.shape.as_list()))
print('Fully connected lyer on all %d rows results in %s outputs:' % (
    my_maxpool_output.shape.as_list()[0], my_full_output.shape.as_list()[0]))
print(sess.run(my_full_output, feed_dict=feed_dict))
