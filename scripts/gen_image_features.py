'''
Sanjay Singh
san.singhsanjay@gmail.com
May-2021
To generate and save InceptionV3 features of images
'''

# packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
import math

# constants
IMG_WIDTH = 299
IMG_HEIGHT = 299
IMG_CHANNEL = 3
BATCH_SIZE = 64

# paths
train_data_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/train_image_caption_processed.csv" 
train_image_source_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_dataset/Images/"
train_target_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_dataset/inception_train_npy_files/"

# function to generate image bottleneck features - from InceptionV3
def gen_save_bottleneck_features(path, imagenames):
	# loading InceptionV3 model
	image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
	new_input = image_model.input
	hidden_layer = image_model.layers[-1].output
	image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
	# loading each image, preprocessing it, generating bottleneck features and saving it
	batch_size = len(imagenames)
	image_batch = np.ndarray([batch_size, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])
	for i in range(len(imagenames)):
		img = tf.io.read_file(path + imagenames[i])
		img = tf.image.decode_jpeg(img, channels=3)
		img = tf.image.resize(img, (IMG_WIDTH, IMG_HEIGHT))
		img = tf.keras.applications.inception_v3.preprocess_input(img)
		image_batch[i] = img
	features = image_features_extract_model(image_batch)
	features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))
	for i in range(features.shape[0]):
		np.save(train_target_path + imagenames[i] + ".npy", features[i].numpy())

# loading data
data_df = pd.read_csv(train_data_path)
imagenames = list(data_df['image'])

# generating and saving inceptionV3 bottleneck features
no_of_steps = math.ceil(len(imagenames) / BATCH_SIZE)
print("Generating InceptionV3 bottleneck features:")
for i in tqdm(range(no_of_steps)):
	start = i * BATCH_SIZE
	end = start + BATCH_SIZE
	gen_save_bottleneck_features(train_image_source_path, imagenames[start : end])
print()
