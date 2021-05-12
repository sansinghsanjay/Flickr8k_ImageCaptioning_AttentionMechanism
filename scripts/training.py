'''
Sanjay Singh
san.singhsanjay@gmail.com
April-2021
Implementation of:
https://github.com/subhamio/image-captioning-using-attention-mechanism-local-attention-and-global-attention-/blob/master/image_captioning_using_attention_mechanism.ipynb
'''

# packages
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
import string
from tqdm import tqdm
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
#from keras.backend.tensorflow_backend import set_session
import keras
import sys, time, os, warnings 
warnings.filterwarnings("ignore")
import re
import numpy as np
import pandas as pd 
from PIL import Image
import pickle
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import shuffle
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.model_selection import train_test_split
from os import listdir

# CONSTANTS
EMBEDDING_DIM = 256
BATCH_SIZE = 32
BUFFER_SIZE = 1000
UNITS = 512

# https://www.tensorflow.org/tutorials/text/image_captioning
class VGG16_Encoder(tf.keras.Model):
	# This encoder passes the features through a Fully connected layer
	def __init__(self, EMBEDDING_DIM):
		super(VGG16_Encoder, self).__init__()
		self.fc = tf.keras.layers.Dense(EMBEDDING_DIM)
		self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)
	def call(self, x):
		x = self.fc(x)
		x = tf.nn.relu(x)
		return x

class Rnn_Local_Decoder(tf.keras.Model):
	def __init__(self, embedding_dim, UNITS, vocab_size):
		super(Rnn_Local_Decoder, self).__init__()
		self.units = UNITS
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True,                              recurrent_initializer='glorot_uniform')
		self.fc1 = tf.keras.layers.Dense(self.units)
		self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)
		self.batchnormalization = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
		self.fc2 = tf.keras.layers.Dense(vocab_size)
		# Implementing Attention Mechanism
		self.Uattn = tf.keras.layers.Dense(UNITS)
		self.Wattn = tf.keras.layers.Dense(UNITS)
		self.Vattn = tf.keras.layers.Dense(1)

	def call(self, x, features, hidden):
		# features shape ==> (64,49,256) ==> Output from ENCODER
		# hidden shape == (batch_size, hidden_size) ==>(64,512)
		# hidden_with_time_axis shape == (batch_size, 1, hidden_size) ==> (64,1,512)
		hidden_with_time_axis = tf.expand_dims(hidden, 1)
		# score shape == (64, 49, 1)
		# Attention Function
		'''e(ij) = f(s(t-1),h(j))'''
		''' e(ij) = Vattn(T)*tanh(Uattn * h(j) + Wattn * s(t))'''
		score = self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)))
		# self.Uattn(features) : (64,49,512)
		# self.Wattn(hidden_with_time_axis) : (64,1,512)
		# tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)) : (64,49,512)
		# self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis))) : (64,49,1) ==> score
		# you get 1 at the last axis because you are applying score to self.Vattn
		# Then find Probability using Softmax
		'''attention_weights(alpha(ij)) = softmax(e(ij))'''
		attention_weights = tf.nn.softmax(score, axis=1)
		# attention_weights shape == (64, 49, 1)
		# Give weights to the different pixels in the image
		''' C(t) = Summation(j=1 to T) (attention_weights * VGG-16 features) '''
		context_vector = attention_weights * features
		context_vector = tf.reduce_sum(context_vector, axis=1)
		# Context Vector(64,256) = AttentionWeights(64,49,1) * features(64,49,256)
		# context_vector shape after sum == (64, 256)
		# x shape after passing through embedding == (64, 1, 256)
		x = self.embedding(x)
		# x shape after concatenation == (64, 1,  512)
		x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
		# passing the concatenated vector to the GRU
		output, state = self.gru(x)
		# shape == (batch_size, max_length, hidden_size)
		x = self.fc1(output)
		# x shape == (batch_size * max_length, hidden_size)
		x = tf.reshape(x, (-1, x.shape[2]))
		# Adding Dropout and BatchNorm Layers
		x= self.dropout(x)
		x= self.batchnormalization(x)
		# output shape == (64 * 512)
		x = self.fc2(x)
		# shape : (64 * 8329(vocab))
		return x, state, attention_weights

	def reset_state(self, batch_size):
		return tf.zeros((batch_size, self.units))

def loss_function(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)
	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask
	return tf.reduce_mean(loss_)

@tf.function
def train_step(img_tensor, target):
	loss = 0
	# initializing the hidden state for each batch
	# because the captions are not related from image to image
	hidden = decoder.reset_state(batch_size=target.shape[0])
	dec_input = tf.expand_dims([wordtoix['startseq']] * BATCH_SIZE, 1)
	with tf.GradientTape() as tape:
		features = encoder(img_tensor)
		for i in range(1, target.shape[1]):
			# passing the features through the decoder
			predictions, hidden, _ = decoder(dec_input, features, hidden)
			loss += loss_function(target[:, i], predictions)
			# using teacher forcing
			dec_input = tf.expand_dims(target[:, i], 1)
	total_loss = (loss / int(target.shape[1]))
	trainable_variables = encoder.trainable_variables + decoder.trainable_variables
	gradients = tape.gradient(loss, trainable_variables)
	optimizer.apply_gradients(zip(gradients, trainable_variables))
	return loss, total_loss

# function to read, resize and preprocess an image
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = preprocess_input(img)
    return img, image_path 

def map_func(img_name, cap):
	img_tensor = np.load(img_name.decode('utf-8')+'.npy')
	return img_tensor, cap

# paths
images_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_dataset/Images/"
img_captions_csv_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/output/intermediate_files/train_image_caption_processed.csv"
vocabulary_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/output/intermediate_files/vocabulary.txt"
max_caption_len_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/output/intermediate_files/max_caption_length.txt"

# reading dataset
data = pd.read_csv(img_captions_csv_path)
img_name = list(data['image'])
img_caption = list(data['caption'])

# appending image names to their paths
for i in range(len(img_name)):
	img_name[i] = images_path + img_name[i]

# reading vocabulary file
vocabulary = list()
f_ptr = open(vocabulary_path, "r")
lines = f_ptr.readlines()
for line in lines:
	vocabulary.append(line.strip())

# finding size of vocabulary
vocab_size = len(vocabulary) + 1 # added 1 for padded zero
print("Vocabulary Size: ", vocab_size)

# making word-to-index and index-to-word dictionary
wordtoix = dict()
ixtoword = dict()
for i in range(len(vocabulary)):
	wordtoix[vocabulary[i]] = (i + 1)
	ixtoword[(i + 1)] = vocabulary[i]

# finding max caption length
f_ptr = open(max_caption_len_path, 'r')
data = f_ptr.read()
f_ptr.close()
max_caption_len = int(data.split(":")[1].strip())
print("Max Caption Length: ", max_caption_len)

# converting captions to their indices
all_img_names = list()
all_captions = list()
img_caption_ix = list()
for i in range(len(img_caption)):
	captions = img_caption[i].split("#")
	for caption in captions:
		all_img_names.append(img_name[i])
		all_captions.append(caption)
		words = caption.split(" ")
		temp_list = list()
		for word in words:
			if(word in wordtoix):
				temp_list.append(wordtoix[word])
		img_caption_ix.append(temp_list)

# padding zeros to each caption to make it equal to max_caption_len
img_caption_padded = tf.keras.preprocessing.sequence.pad_sequences(img_caption_ix, max_caption_len, padding='post')

# replicating each value of img_name 5 times to make it equal to the length of img_caption_padded
img_name_padded = list()
for i in range(len(img_name)):
	for j in range(5):
		img_name_padded.append(img_name[i])
print("Shape of img_name_padded: ", len(img_name_padded))

# loading vgg-16 model to extract bottleneck features
image_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

'''
# to map data to loading function - to create npy files
encode_train = sorted(set(img_name))
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)

# saving npy files
for img, path in tqdm(image_dataset):
	batch_features = image_features_extract_model(img) # batch_features.shape: [BATCH_SIZE, 7, 7, 512]
	batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3])) # batch_features.shape: [BATCH_SIZE, 49, 512]
	for bf, p in zip(batch_features, path):
		path_of_feature = p.numpy().decode("utf-8")
		np.save(path_of_feature, bf.numpy())
'''

# defining optimization and loss-object
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# making image and caption map - for training
dataset = tf.data.Dataset.from_tensor_slices((img_name_padded, img_caption_padded))
dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# defining encoder and decoder
encoder = VGG16_Encoder(EMBEDDING_DIM)
decoder = Rnn_Local_Decoder(EMBEDDING_DIM, UNITS, vocab_size)

# defining num_steps 
num_steps = len(img_name_padded) // BATCH_SIZE

# training related parameters and training
loss_plot = []
start_epoch = 0
EPOCHS = 20
for epoch in range(start_epoch, EPOCHS):
	start = time.time()
	total_loss = 0
	for (batch, (img_tensor, target)) in enumerate(dataset):
		#target = tf.reshape(target, (1, target.shape[0]))
		batch_loss, t_loss = train_step(img_tensor, target)
		total_loss += t_loss
		if batch % 100 == 0:
			print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
	# storing the epoch end loss value to plot later
	loss_plot.append(total_loss / num_steps)
	print ('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss/num_steps))
	print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
