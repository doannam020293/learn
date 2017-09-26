import numpy as np
import tensorflow as tf
from sklearn.datasets import load_sample_images
import cv2
import matplotlib.pyplot as plt
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "cnn"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

def plot_color_image(image):
    plt.imshow(image.astype(np.uint8),interpolation="nearest")
    plt.axis("off")





dataset = np.array(load_sample_images().images,np.float32)
# cv2.imshow('1',dataset[1]) # show ảnh.

batch_size, height, width, channel = dataset.shape

filter_test = np.zeros((7,7,channel,2),dtype=np.float32)
filter_test[:,3,:,1] = 1
# filter_test[3,:,:,0] = 1


X = tf.placeholder(tf.float32,(batch_size,height,width,channel), name='X')
convolution = tf.nn.conv2d(X,filter_test,strides =[1,1,1,1],padding= 'SAME')
max_pool = tf.nn.max_pool(X,strides=[1,2,2,1],padding="VALID",ksize=[1,2,2,1])
with tf.Session() as sess:
    # out =sess.run(convolution,feed_dict={X:dataset})
    out1 =sess.run(max_pool,feed_dict={X:dataset})


plt.figure()

plot_color_image(dataset[0])
plot_color_image(out1[0])

plt.imshow(out[0, :, :, 1])
plt.imshow(out[0, :, :, 0])
plt.imshow(out[1, :, :, 0])

plt.show()


