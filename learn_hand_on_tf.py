import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np




x = tf.Variable(3)
y = tf.Variable(4)

f = x*x*y + y +2

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    r = f.eval()

graph = tf.Graph()

with graph.as_default():
    x2 = tf.Variable(3)

x2.graph is graph
x2.graph is tf.get_default_graph()


a = tf.constant(3.0)
b = tf.constant(4.0)

c = tf.add(a,b)

sess = tf.Session()
sess.run(c)


a1 = tf.placeholder(tf.float32)
b1 = tf.placeholder(tf.float32)
c1 = a1 + b1
d1 = c1**2
sess.run(d1,{b1:1, a1:5})

W = tf.Variable([3],dtype= tf.float32)
b = tf.Variable([1],dtype= tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
init = tf.global_variables_initializer()
sess.run(init)
sess.run(linear_model,{x:[10,11,12,13]})
y = tf.placeholder(tf.float32)
square = tf.square(y-linear_model)
sum_square = tf.reduce_sum(square)
square_root = tf.sqrt(sum_square)
size = tf.size(y)

sess.run(size)

mean_error = square_root/size
sess.run(mean_error,{y:[32,33,36,39]})

tensor2 = tf.reshape(y, tf.TensorShape([-1, 1]))
tensor.get_shape().as_list()


