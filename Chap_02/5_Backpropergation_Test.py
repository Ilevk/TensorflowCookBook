import numpy as np
import tensorflow as tf

sess = tf.Session()

# x_values of normal distribution mean 1 standard derivation 0.1
x_vals = np.random.normal(1, 0.1, 100)
# all about answer is 10
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1]))
# L2_norm
my_output = tf.multiply(x_data, A)

loss = tf.square(my_output - y_target)
# must initialize variables
init = tf.global_variables_initializer()
sess.run(init)
# use GradientDescent Method to minimize loss function
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)
print('Regression Test')
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1) % 25 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

print('\n\n'
      'Classification Test')
tf.reset_default_graph()
sess = tf.Session()
x_vals = np.concatenate((np.random.normal(-1, 1, 50),
                         np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
#A have a gap(10) from target(-1) so we can see how to move loss func
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

my_output = tf.add(x_data, A)

my_output_expanded = tf.expand_dims(my_output, 0)
y_target_expanded = tf.expand_dims(y_target, 0)

init = tf.global_variables_initializer()
sess.run(init)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
    logits=my_output_expanded, labels=y_target_expanded
)
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

for i in range(1400):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1) % 200 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy,
                                       feed_dict={x_data: rand_x, y_target: rand_y})))
