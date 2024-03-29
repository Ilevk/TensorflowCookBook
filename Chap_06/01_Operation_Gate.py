import tensorflow as tf

sess = tf.Session()
#Define Variable
a = tf.Variable(tf.constant(4.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)
#ax gate
multiplication = tf.multiply(a, x_data)
#Square error
loss = tf.square(tf.subtract(multiplication, 50.))

init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

print('Optimizing a Multiplication Gate Output to 50.')
for i in range(10):
    sess.run(train_step, feed_dict={x_data: x_val})
    a_vals = sess.run(a)
    mult_output = sess.run(multiplication, feed_dict={x_data: x_val})
    print(str(a_vals) + ' * ' + str(x_val) + ' = ' + str(mult_output))

from tensorflow.python.framework import ops

ops.reset_default_graph()
sess = tf.Session()
#Add Additional Variable
a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)

# ax + b gate
two_gate = tf.add(tf.multiply(a, x_data), b)
loss = tf.square(tf.subtract(two_gate, 50.))

init = tf.global_variables_initializer()
sess.run(init)
# Use GradientDescent Method
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

print('\nOptimizing Two Gate Output to 50.')
for i in range(10):
    sess.run(train_step, feed_dict={x_data: x_val})
    a_val, b_val = (sess.run(a), sess.run(b))
    two_gate_output = sess.run(two_gate, feed_dict = {x_data: x_val})
    print(str(a_val) + ' * ' + str(x_val) + ' + ' + str(b_val) + ' = '+ str(two_gate_output))
