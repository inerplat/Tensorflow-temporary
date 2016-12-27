import tensorflow as tf

hello = tf.constant('Hello, tensorflow!')
sses = tf.Session()
print (hello)
print(sses.run(hello))

a = tf.constant(10)
b = tf.constant(32)
c=a+b
print (c)
print(sses.run(c))