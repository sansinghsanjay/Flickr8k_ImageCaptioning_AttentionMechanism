'''
Sanjay Singh
san.singhsanjay@gmail.com
April-2021
To generate and save bottleneck features of images - model used Invecption V3
'''

# packages
import pandas as pd
import cv2
import numpy as np
from keras.models import Sequential
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model

# constants
IMG_WIDTH = 299
IMG_HEIGHT = 299
IMG_CHANNEL = 3

# function to update status
def percentage_progress(completed, total):
	perc_progress = (completed / total) * 100
	perc_prgoress = round(perc_progress, 2)
	return perc_progress

# function to generate image bottleneck features - from InceptionV3
def gen_bottleneck_features(train_filenames, from_ix, to_ix):
	# read images, resize them and preprocess them according to inception_v3
	train_images = np.ndarray([len(train_filenames), IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])
	for i in range(len(train_filenames)):
		img = cv2.imread(image_path + train_filenames[i])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
		img = preprocess_input(img)
		train_images[i] = img
		if((i + 1) % 500 == 0 or (i + 1) == len(train_filenames)):
			perc_progress = percentage_progress((i + 1), len(train_filenames))
			print("Completed reading and pre-processing of images: ", perc_progress, " %", end='\r')
	print()	
	# instantiating model
	model = InceptionV3(weights='imagenet')
	# removing last layer (output layer) of inception_v3 model
	model = Model(model.input, model.layers[-2].output)
	# generating bottleneck feature of images
	print("Generating bottleneck features of all train images by using InceptionV3. Wait...")
	feat_vec = model.predict(train_images)
	print("Generated bottleneck features for all train images")
	# converting np.ndarray type feat_vec into pd.DataFrame
	feat_vec = pd.DataFrame(feat_vec)
	# adding column at beginning for image name
	feat_vec.insert(loc=0, column='image', value=train_filenames)
	# saving feat_vec
	filename = "gen_image_vec_" + str(from_ix) + "_" + str(to_ix) + ".csv"
	feat_vec.to_csv(target_path + filename, index=False)
	print("Dimension of generated csv (containing image filenames and bottleneck features): ", feat_vec.shape)
	print("Saved ", filename)

# path
image_path = "/home/sansingh/Downloads/Flickr8k_ImageCaptioning/archive/Images/"
train_filenames_path = "/home/sansingh/Downloads/Flickr8k_ImageCaptioning/archive/Flickr_8k.trainImages.txt"
target_path = "/home/sansingh/Downloads/Flickr8k_ImageCaptioning/output/intermediate_files/"

# reading train_filenames
train_filenames = list()
f_ptr = open(train_filenames_path, "r")
lines = f_ptr.readlines()
for line in lines:
	train_filenames.append(line.strip())
print("Completed reading training filenames")

# due to memory issues, generating bottleneck features and saving them in parts
gen_bottleneck_features(train_filenames[0:1000], 1, 1000)
gen_bottleneck_features(train_filenames[1000:2000], 1001, 2000)
gen_bottleneck_features(train_filenames[2000:3000], 2001, 3000)
gen_bottleneck_features(train_filenames[3000:4000], 3001, 4000)
gen_bottleneck_features(train_filenames[4000:5000], 4001, 5000)
gen_bottleneck_features(train_filenames[5000:6000], 5001, 6000)
