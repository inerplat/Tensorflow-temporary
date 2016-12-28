import tensorflow as tf


x_data = [1., 2., 3., 4.]
y_data = [3., 5., 7., 9.]


W = tf.Variable(tf.random_uniform([1], -10000., 10000.))
b = tf.Variable(tf.random_uniform([1], -10000., 10000.))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

H = W * X + b

cost = tf.reduce_mean(tf.square(H - Y))


alpha = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()


sess = tf.Session()
sess.run(init)


for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print (step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))

print (sess.run(H, feed_dict={X: [5, 2.5, 7, 3.5]}))