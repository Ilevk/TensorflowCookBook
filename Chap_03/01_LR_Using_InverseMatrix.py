import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#Define Variables
sess = tf.Session()
x_vals = np.linspace(0,10,100)
y_vals = x_vals + np.random.normal(0,1,100)
#We can present LR like Ax = b. if A weren't Square Matrix,we need to give attention
#x equals (At*A)^(-1)*(At)*b
x_vals_column =np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1,100)))
A = np.column_stack((x_vals_column, ones_column))
b = np.transpose(np.matrix(y_vals))

A_tensor = tf.constant(A)
b_tensor = tf.constant(b)
#At*A
tA_A = tf.matmul(tf.transpose(A_tensor),A_tensor)
#(At*A)^(-1)
tA_A_inv = tf.matrix_inverse(tA_A)
#(At*A)^(-1)*(At)
product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
#(At*A)^(-1)*(At)*b
solution = tf.matmul(product, b_tensor)
#result
solution_eval = sess.run(solution)

slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]
print('slope : ' + str(slope))
print('y_intercept: ' + str(y_intercept))

#Get best fitted results
best_fit = []
for i in x_vals:
    best_fit.append(slope*i+y_intercept)
#plotting
plt.plot(x_vals,y_vals, 'o', label = 'Data')
plt.plot(x_vals,best_fit, 'r-', label='Best fit line')
plt.legend(loc='upper left')
plt.show()