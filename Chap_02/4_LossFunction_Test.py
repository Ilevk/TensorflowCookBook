import matplotlib.pyplot as plt
import tensorflow as tf

sess = tf.Session()

x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)

# L2 norm func has a big gradient around target -> will be good loss func
l2_y_vals = tf.square(target - x_vals)
l2_y_out = sess.run(l2_y_vals)

# L1 norm func less respond target - value -> process well than L2
l1_y_vals = tf.abs(target - x_vals)
l1_y_out = sess.run(l1_y_vals)

# pseudo-Huber is approximation of the huber function. this has less sharp form far target and convex form near target.
# delta value will choose gradient degree
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals) / delta1)) - 1.)
phuber1_y_out = sess.run(phuber1_y_vals)

delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2),tf.sqrt(1.+tf.square((target-x_vals)/delta2)) - 1.)
phuber2_y_out = sess.run(phuber2_y_vals)

x_vals = tf.linspace(-3.,5.,500)
target = tf.constant(1.)
targets = tf.fill([500,],1.)

#hinge not only use SVM but also Neural Network.
hinge_y_vals = tf.maximum(0.,1. - tf.multiply(target, x_vals))
hinge_y_out = sess.run(hinge_y_vals);

#cross-entropy called logistic loss function is used to predict 0 or 1 classification
#(exactly predict a distance to predicted result from true clossification result
xentropy_y_vals = tf.multiply(target, tf.log(x_vals)) - tf.multiply((1. - target),tf.log(1. - x_vals))
xentropy_y_out = sess.run(xentropy_y_vals)

#sigmoid cross entropy is almost same with cross-entropy func. Difference is just to transpose x_value to sigmoid before x value input to loss func
x_vals_input = tf.expand_dims(x_vals,1)
target_input = tf.expand_dims(targets,1)
xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_input, logits=x_vals_input)
xentropy_sigmoid_y_out = sess.run(xentropy_y_vals)

#weighted cross entropy func is adding weight to sigmoid cross entropy function
weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(targets,x_vals, weight)
xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)

#softmax cross entropy is targeting unnormalized output values.
#this use to estimate non-various classification target just one target. So that can be transposed to probability
unscaled_logits = tf.constant([[1.,-3.,10.]])
target_dist = tf.constant([[.1,.02,.88]])
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=unscaled_logits, labels = target_dist)
print(sess.run(softmax_xentropy))

#sparse sofrmax cross entropy is targeting where is exact class
unscaled_logits = tf.constant([[1.,-3.,10.]])
sparse_target_dist = tf.constant([2])
sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = unscaled_logits,labels=sparse_target_dist)
print(sess.run(sparse_xentropy))