import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime



def neural_network(X, shape_output, name_scope, activate_fn=None):
    shape_input = X.get_shape()[1]
    # tạo standard derivation khi khởi tạo W theo phân phối Gaussian sẽ giúp converge nhanh hơn khi optimization.
    stddev = 2 / np.sqrt(shape_input)

    with tf.name_scope(name_scope):
        W = tf.Variable(tf.truncated_normal([shape_input, shape_output], stddev=stddev), name='W')
        b = tf.Variable(np.zeros([shape_output]),
                        name='b')  # b thì khởi tạo = 0 cũng được, không có vấn đề khi optimization
        z = tf.matmul(X, W) + b
        if activate_fn == 'relu':
            y = tf.nn.relu(z)
        else:
            y = z
    return y

mnist = input_data.read_data_sets('/tmp/data/')

n_inputs = 28* 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y = tf.placeholder(tf.int64,shape=(None),name="y")


with tf.name_scope('dnn'):
    hidden1 = fully_connected(X, n_hidden1, scope='hidden1',)
    hidden2 = fully_connected(hidden1, n_hidden2, scope='hidden2')
    output = fully_connected(hidden2, n_outputs, scope='output', activation_fn=None)
    # ta k dùng active function là sofmax ngay, do ta sẽ dungf sparse_softmax_cross_entropy_with_logits, function này sẽ apply softmax cho ta, và care những case đặc biệt như logit của 0

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
                                                              labels=y)  # label y laf 1D tensor với giá trị tự 0, đến số classes - 1 (ví dụ tự 0 đến 9)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01
with  tf.name_scope('train'):
    optimiser = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimiser.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(output, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

now  =datetime.now().strftime("%Y%m%d%H%M%S")
# root_logdir = r'C:\nam\work\tf_log'
root_logdir = 'tf_log'
logdir = '{}/run_{}'.format(root_logdir,now)
accuracy_summary = tf.summary.scalar('accuracy',accuracy)
summary_op = tf.summary.tensor_summary('softmax_input', output)
file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())


n_epochs = 1
batch_size = 50
model_path =  './my_model_temp.ckpt'
n_step_display = 250
with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, model_path)
    for epoch in range(n_epochs):
        internal_loop  = mnist.train.num_examples//batch_size
        for batch_index in range(internal_loop):
            input_train= mnist.train.next_batch(batch_size)
            _, out_eval =  sess.run([training_op,output],feed_dict={X:input_train[0],y:input_train[1]})
            print(input_train[1])
            print(out_eval)
            print('-------------------')
            if batch_index % n_step_display==0:
                # saver.save(sess,model_path)
                # accuracy_eval = sess.run(accuracy_summary, feed_dict={X: input_train[0], y: input_train[1]})
                accuracy_test = sess.run(accuracy_summary, feed_dict={X: mnist.test.images, y: mnist.test.labels})
                summary_op_eval = sess.run(summary_op, feed_dict={X: mnist.test.images})
                step = epoch*internal_loop + batch_index
                # file_writer.add_summary(accuracy_eval,step)
                file_writer.add_summary(accuracy_test,step)
                file_writer.add_summary(summary_op_eval,step)

    # saver.save(sess, './my_model_final.ckpt')
file_writer.close()