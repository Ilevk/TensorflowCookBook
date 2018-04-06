import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sess = tf.Session()
# Define x, y values, and placeholder shape [None,1] make easy MatMultiply
batch_size = 20
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
# Make train indices by random select
train_indices = np.random.choice(len(x_vals),
                                 round(len(x_vals) * 0.8), replace=False)
# Make test indices by all indices - train indices
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
A = tf.Variable(tf.random_normal(shape=[1, 1]))

my_output = tf.matmul(x_data, A)
# Define Loss func
loss = tf.reduce_mean(tf.square(my_output - y_target))
# Use GradientDescent Method to minimize loss func
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)
# Init Variables
init = tf.global_variables_initializer()
sess.run(init)

# Train roop
for i in range(100):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1) % 25 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

# To Evaluate Model print MSE Value
mse_test = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
mse_train = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})
print('MSE on test:' + str(np.round(mse_test, 2)))
print('MSE on train:' + str(np.round(mse_train, 2)))
print('\n')

# Now We make accuracy function to use at last. sigmoid function is in loss function so to check Classification well
# We need another sigmoid function
from tensorflow.python.framework import ops

ops.reset_default_graph()
sess = tf.Session()

# Define values and placeholder
batch_size = 25
# Make x value concatenate normal distributions mean -1, variance 1 and mean 2, variance 1
x_vals = np.concatenate((np.random.normal(-1, 1, 50),
                         np.random.normal(2, 1, 50)))
# Make answer
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(shape=[1, None], dtype=tf.float32)
y_target = tf.placeholder(shape=[1, None], dtype=tf.float32)
# Make train indices 80% of all
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
# Make test indices 20% of all
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
# split values
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))
my_output = tf.add(x_data, A)
# Use Loss func sigmoid cross entropy func
xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target))
# Use GradientDescentent Method to minimize loss func
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)
# init globals variables
init = tf.global_variables_initializer()
sess.run(init)

# Print result
for i in range(1800):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = [x_vals_train[rand_index]]
    rand_y = [y_vals_train[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1) % 200 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))

# Define Additional for Evaluate model
y_prediction = tf.squeeze(tf.round(tf.nn.sigmoid(tf.add(x_data, A))))
# Compare prediction with correct answer
correct_prediction = tf.equal(y_prediction, y_target)
# cast boolean value to float value
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_value_test = sess.run(accuracy, feed_dict={x_data: [x_vals_test], y_target: [y_vals_test]})
acc_value_train = sess.run(accuracy, feed_dict={x_data: [x_vals_train], y_target: [y_vals_train]})
print('Accuracy on train set: ' + str(acc_value_train))
print('Accuracy on test set: ' + str(acc_value_test))
