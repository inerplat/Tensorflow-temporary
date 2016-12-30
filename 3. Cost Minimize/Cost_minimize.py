import tensorflow as tf
import matplotlib.pyplot as plt
X = [1., 2., 3.]
Y = [2., 7., 9.]
m = n_samples = len(X)

W = tf.placeholder(tf.float32)

H = tf.mul(X,W)

cost = tf.reduce_sum(tf.pow(H-Y,2))/(m)

init = tf.initialize_all_variables()

W_val = []
cost_val = []

sess = tf.Session()
sess.run(init)

alpha=0.1

for i in range(-30,50):
    print (i*alpha, sess.run(cost, feed_dict={W : i*alpha}))
    W_val.append(i*0.1)
    cost_val.append(sess.                                                                                       run(cost, feed_dict={W : i*alpha}))


plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()