import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path')
parser.add_argument('-m','--model')
parser.add_argument('-l','--labels')
args = parser.parse_args()

labels = args.labels.split(',')

print(labels)

# First, pass the path of the image
image_path = args.path

model_name = args.model
#label1 = 'lingerie'
#label2 = 'not-lingerie'
num_labels = len(labels)

#filename = dir_path +'/' +image_path
image_size = 128
num_channels = 3
images = []
#num_labels = len(os.listdir('training_data'))

# Let us restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph(model_name + '.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

# Let's feed the images to the input placeholders
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, num_labels))

for label in labels:
    try:
        os.mkdir(image_path + '/' + label)
    except:
        pass

files = glob.glob(image_path + '/*.jpg')
files = files + glob.glob(image_path + '/*.jpeg')

for file in files:
    try:
        print(file)
        images = []
        # Reading the image using OpenCV
        image = cv2.imread(file)
        # Resizing the image to our desired size and preprocessing will be done exactly as done during training
        image = cv2.resize(image, (image_size, image_size),
                           0, 0, cv2.INTER_LINEAR)
        images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0/255.0)
        # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
        x_batch = images.reshape(1, image_size, image_size, num_channels)

        # Creating the feed_dict that is required to be fed to calculate y_pred
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result = sess.run(y_pred, feed_dict=feed_dict_testing)
        # result is of this format [probabiliy_of_rose probability_of_sunflower]
        c = 0
        values = []

        for label in labels:
            print(label, result[0, c])
            values.append(result[0, c])
            c += 1
        
        pred = max(values)
        pos = values.index(pred)
        winner = labels[pos]

        #print(values, pred, pos, winner)
        print(os.path.basename(file), '->', winner)

        newfile = image_path + '/' + winner + '/' + os.path.basename(file)
        os.rename(file, newfile)

    except Exception as e:
        print(e, 'Could not load', file)
