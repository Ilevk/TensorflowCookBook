import tensorflow as tf
import numpy as np

sess = tf.Session()

# [ # of images, Height of image, Width of image, # of Channel ]
x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)

x_data = tf.placeholder(tf.float32, shape=x_shape)

#[ Height of filter, Width of filter, input_channel, output_channel ]
my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
# Must have strides[0] = strides[3] = 1, strides[1] is Horizontal, strides[2] is Vertical strides
my_strides = [1, 2, 2, 1]

mov_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving_Abg_Window')

def custom_layer(input_matrix):
    input_matrix_sqeezed = tf.squeeze(input_matrix) #squeeze func will delete Dimensions valued 1
    A = tf.constant([[1.,2.],[-1.,3.]])
    b = tf.constant(1., shape=[2,2])
    temp1 = tf.matmul(A, input_matrix_sqeezed) #temp1 = Ax
    temp = tf.add(temp1,b) #Ax + b
    return(tf.sigmoid(temp))

with tf.name_scope('Custom_Layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)

print(sess.run(custom_layer1, feed_dict={x_data: x_val}))
