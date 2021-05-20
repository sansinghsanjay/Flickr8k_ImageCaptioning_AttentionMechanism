'''
Sanjay Singh
san.singhsanjay@gmail.com
May-2021
Script to generate VGG-16 features of images
'''

# packages
from tqdm import tqdm
import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16, preprocess_input
import tensorflow as tf
import cv2

# constants
IMG_W = 224
IMG_H = 224
BATCH_SIZE = 1

# paths
train_data_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/train_image_caption_processed.csv" 
train_image_source_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_dataset/Images/"
train_target_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_dataset/train_npy_files/" 

# function to generate VGG-16 features and saving them in npy files
def gen_npy_save(image_features_extract_model, source_path, imagenames, target_path, status):
	print("Generating VGG-16 bottleneck features for " + status + " data:")
	filecount = 0
	for imagename in tqdm(imagenames):
		img = cv2.imread(source_path + imagename)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (IMG_W, IMG_H))
		img = preprocess_input(img)
		img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
		img_features = image_features_extract_model(img) # batch_features.shape: [BATCH_SIZE, 7, 7, 512]
		img_features = tf.reshape(img_features, (img.shape[0], -1, img_features.shape[3])) # batch_features.shape: [BATCH_SIZE, 49, 512]
		img_features = img_features.numpy()
		np.save(target_path + imagename + ".npy", img_features[0])
	print()

# reading csv files
train_df = pd.read_csv(train_data_path)

# loading VGG-16
image_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# generating npy files and saving in respecting target path
gen_npy_save(image_features_extract_model, train_image_source_path, list(train_df['image']), train_target_path, "train")
