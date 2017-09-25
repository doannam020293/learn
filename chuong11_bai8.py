import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import  fully_connected
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import dropout
import pandas as pd
import numpy as np
from datetime import  datetime

def filter_input(X,y):
    df = pd.DataFrame({"x":list(X),'y':y})
    df_new = df[df['y']<5]
    x_new, y_new = df_new['x'].values, df_new['y'].values
    x_new = np.array([a for a in x_new])
    return x_new, y_new


mnist = input_data.read_data_sets('/tmp/data/',)
mnis_filter = mnist
n_unit = 100
n_input = 28*28
n_output = 5
n_epochs  = 100
batch_size = 50
n_step_display = 100
n_step_for_early_stop = 1000
keep_prob = 0.8

X =  tf.placeholder(tf.float32, (None,n_input),'X')
y =  tf.placeholder(tf.int32, (None),'y')
is_training = tf.placeholder(tf.bool,name='is_training',shape=())
#scale
bn_param = {
    'is_training': is_training,
    'decay':0.999,
    'updates_collections':None,
    'scale':True

}

with tf.name_scope('dnn'):
    # tao list dict cho full_connection
    with tf.contrib.framework.arg_scope(
        [fully_connected],
        normalizer_fn= batch_norm,
        normalizer_params= bn_param,
        weights_initializer=variance_scaling_initializer()

    ):
        X_drop = dropout(X,keep_prob=keep_prob,is_training=is_training)
        h1 = fully_connected(X_drop,n_unit,activation_fn=tf.nn.elu,scope='h1')
        h1_drop = dropout(h1, keep_prob=keep_prob, is_training=is_training)
        h2 = fully_connected(h1_drop,n_unit,activation_fn=tf.nn.elu,scope='h2')
        h2_drop = dropout(h2, keep_prob=keep_prob, is_training=is_training)
        h3 = fully_connected(h2_drop,n_unit,activation_fn=tf.nn.elu,scope='h3')
        h3_drop = dropout(h3, keep_prob=keep_prob, is_training=is_training)
        h4 = fully_connected(h3_drop,n_unit,activation_fn=tf.nn.elu,scope='h4')
        h4_drop = dropout(h4, keep_prob=keep_prob, is_training=is_training)
        h5 = fully_connected(h4_drop,n_unit,activation_fn=tf.nn.elu,scope='h5')
        h5_drop = dropout(h5, keep_prob=keep_prob, is_training=is_training)
        logit = fully_connected(h5_drop+
                                ,n_output,activation_fn=None,scope='output')
        # logit la 1 tensor với shape (n_batches, n_output) đây là 1 số chưa qua hàm softmax, nhưng dựa vào vị trí có giá trị lớn nhất thì ta có thể biết được neural network của ta đang dự đoán là số mấy

with tf.name_scope('loss'):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logit)

with tf.name_scope('train'):
    optimisor = tf.train.AdamOptimizer()
    train_op = optimisor.minimize(loss)

with tf.name_scope('eval'):
    accuracy_raw = tf.nn.in_top_k(predictions=logit,targets=y,name= 'in_top_k',k = 1)
    #  kiểm tra  target có nằm trong top k của prediction hay không, return True False array
    accuracy = tf.reduce_mean(tf.cast(accuracy_raw,tf.float32), name='accuracy',)
    accuracy1 = tf.reduce_mean(tf.cast(accuracy_raw,tf.float32), name='accuracy1',)

with tf.name_scope('summary'):
    now = datetime.now()
    root_logdir = 'tf_log'
    log_dir = '{}/run_{}'.format(root_logdir, now)
    accuracy_summary = tf.summary.scalar(tensor= accuracy,name='accuracy_summary')
    accuracy_summary_test = tf.summary.scalar(tensor= accuracy1,name='accuracy_summary_test')
    file_writer = tf.summary.FileWriter(log_dir,tf.get_default_graph())
with tf.name_scope('save'):
    saver = tf.train.Saver()


# phanze operation
with tf.Session() as sess:
    best_accuracy_test = 0
    step_for_early_stop = 0
    init = tf.global_variables_initializer()  # variable đưuóc khởi tạo trong hàm fully_connected
    x_shape = mnist.train.num_examples
    n_batch = int(x_shape // batch_size)

    sess.run(init)
    for epoch in range(n_epochs):

        for step in range(n_batch):
            X_input, y_input = mnist.train.next_batch(batch_size)
            X_filter, y_filter    = filter_input(X_input, y_input)
            sess.run(train_op,feed_dict={X: X_filter, y: y_filter,is_training:True})
            if step %n_step_display  ==0:
                X_test, y_test = mnist.test.images, mnist.test.labels
                X_test_filter, y_test_filter = filter_input(X_test, y_test )
                accuracy_summary_eval = sess.run(accuracy_summary, feed_dict={X: X_filter, y: y_filter,is_training:True})
                accuracy_summary_test_eval = sess.run(accuracy_summary_test, feed_dict={X: X_test_filter, y: y_test_filter,is_training:False})
                accuracy_test_eval = sess.run(accuracy, feed_dict={X: X_test_filter, y: y_test_filter,is_training:False})
                if accuracy_test_eval > best_accuracy_test:
                    best_accuracy_test = accuracy_test_eval
                    step_for_early_stop +=1
                    if step_for_early_stop ==n_step_for_early_stop:
                        print('early stop')
                        saver.save(sess, 'final.ckpt')
                        break
                global_step = epoch*batch_size + step
                file_writer.add_summary(accuracy_summary_eval,global_step)
                file_writer.add_summary(accuracy_summary_test_eval,global_step)
                print(accuracy_summary_eval)
    saver.save(sess,'final.ckpt')

file_writer.close()








