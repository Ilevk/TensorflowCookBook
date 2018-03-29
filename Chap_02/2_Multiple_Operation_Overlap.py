import tensorflow as tf
import numpy as np

sess = tf.Session()

#Define 3x5 numpy array
my_array = np.array([[1.,3.,5.,7.,9.],
                     [-2.,0.,2.,4.,6.],
                    [-6.,-3.,0.,3.,6.]])
x_vals = np.array([my_array, my_array+1])
x_data = tf.placeholder(tf.float32, shape=(3,5))

m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

#multiply 3x5 and 5x1 arrays result will be 3x1 array
prod1 = tf.matmul(x_data,m1)
#multiply 3x1 and 1x1 arrays result will be 3x1 array
prod2 = tf.matmul(prod1,m2)
#just add constant to 3x1 array
add1 = tf.add(prod2,a1)

for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data : x_val}))