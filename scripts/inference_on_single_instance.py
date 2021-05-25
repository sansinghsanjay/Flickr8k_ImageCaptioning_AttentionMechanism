'''
Sanjay Singh
san.singhsanjay@gmail.com
May-2021
To make inference for Image Captioning by using Visual Attention Mechanism
Followed this:
https://www.tensorflow.org/tutorials/text/image_captioning
'''

img_path = ["/home/sansingh/Downloads/man_cleaning_floor.jpg"]

# packages
import tensorflow as tf
import random
import numpy as np
import os
import time
from tqdm import tqdm
import pandas as pd

# constants
IMG_WIDTH = 299
IMG_HEIGHT = 299
IMG_CHANNEL = 3
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIMS = 256
UNITS = 512
EPOCHS = 20

# paths
vocabulary_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/vocabulary.txt"
max_caption_len_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/max_caption_length.txt"
checkpoint_path = "/home/sansingh/github_repo/trained_models/Flickr8k_ImageCaptioning_AttentionMechanism/ckpt-1"

class BahdanauAttention(tf.keras.Model):
	def __init__(self, units):
		super(BahdanauAttention, self).__init__()
		self.W1 = tf.keras.layers.Dense(units)
		self.W2 = tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)

	def call(self, features, hidden):
		# features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
		# hidden shape == (batch_size, hidden_size)
		# hidden_with_time_axis shape == (batch_size, 1, hidden_size)
		hidden_with_time_axis = tf.expand_dims(hidden, 1)
		# attention_hidden_layer shape == (batch_size, 64, units)
		attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
		# score shape == (batch_size, 64, 1)
		# This gives you an unnormalized score for each image feature.
		score = self.V(attention_hidden_layer)
		# attention_weights shape == (batch_size, 64, 1)
		attention_weights = tf.nn.softmax(score, axis=1)
		# context_vector shape after sum == (batch_size, hidden_size)
		context_vector = attention_weights * features
		context_vector = tf.reduce_sum(context_vector, axis=1)
		return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
	# Since you have already extracted the features and dumped it
	# This encoder passes those features through a Fully connected layer
	def __init__(self, embedding_dim):
		super(CNN_Encoder, self).__init__()
		# shape after fc == (batch_size, 64, embedding_dim)
		self.fc = tf.keras.layers.Dense(embedding_dim)

	def call(self, x):
		x = self.fc(x)
		x = tf.nn.relu(x)
		return x

class RNN_Decoder(tf.keras.Model):
	def __init__(self, embedding_dim, units, vocab_size):
		super(RNN_Decoder, self).__init__()
		self.units = units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True,              recurrent_initializer='glorot_uniform')
		self.fc1 = tf.keras.layers.Dense(self.units)
		self.fc2 = tf.keras.layers.Dense(vocab_size)
		self.attention = BahdanauAttention(self.units)

	def call(self, x, features, hidden):
		# defining attention as a separate model
		context_vector, attention_weights = self.attention(features, hidden)
		# x shape after passing through embedding == (batch_size, 1, embedding_dim)
		x = self.embedding(x)
		# x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
		x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
		# passing the concatenated vector to the GRU
		output, state = self.gru(x)
		# shape == (batch_size, max_length, hidden_size)
		x = self.fc1(output)
		# x shape == (batch_size * max_length, hidden_size)
		x = tf.reshape(x, (-1, x.shape[2]))
		# output shape == (batch_size * max_length, vocab)
		x = self.fc2(x)
		return x, state, attention_weights

	def reset_state(self, batch_size):
		return tf.zeros((batch_size, self.units))

def loss_function(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)
	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask
	return tf.reduce_mean(loss_)

# function to load image
def load_image(path):
	image_batch = np.ndarray([len(path), IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])
	img = tf.io.read_file(path[0])
	img = tf.image.decode_jpeg(img, channels=3)
	img = tf.image.resize(img, (IMG_WIDTH, IMG_HEIGHT))
	img = tf.keras.applications.inception_v3.preprocess_input(img)
	image_batch[0] = img
	return image_batch

# reading vocabulary file
vocabulary = list()
f_ptr = open(vocabulary_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
for line in lines:
	vocabulary.append(line.strip())
vocab_size = len(vocabulary) + 1 # 1 is added for zero
print("Successfully loaded data from vocabulary file")
print("Vocabulary Size (after adding 1 for padding 0): ", vocab_size)

# creating wordtoix dictionary
ixtoword = dict()
wordtoix = dict()
for i in range(len(vocabulary)):
	wordtoix[vocabulary[i]] = i
	ixtoword[i] = vocabulary[i]
print("Successfully created wordtoix dictionary")

# reading maximum caption length
f_ptr = open(max_caption_len_path, 'r')
data = f_ptr.read()
f_ptr.close()
max_caption_len = int(data.split(":")[1].strip())
print("Max Caption Length: ", max_caption_len)

# loading InceptionV3 model
image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# loading image, generating vgg-16 features 
temp_input = tf.expand_dims(load_image(img_path)[0], 0)
img_tensor_val = image_features_extract_model(temp_input)
img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

# optimizer and loss function
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# making object of encoder and decoder
encoder = CNN_Encoder(EMBEDDING_DIMS)
decoder = RNN_Decoder(EMBEDDING_DIMS, UNITS, vocab_size)

# creating checkpoint for model
ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
ckpt.restore(checkpoint_path)

# generating encoder features
features = encoder(img_tensor_val)
dec_input = tf.expand_dims([wordtoix['<startseq>']], 0)
result = []

# generating result
hidden = decoder.reset_state(batch_size=1)
for i in range(max_caption_len):
	predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
	#attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
	predicted_id = tf.argmax(predictions[0]).numpy()
	result.append(ixtoword[predicted_id])
	if(ixtoword[predicted_id] == '<endseq>'):
		break
	dec_input = tf.expand_dims([predicted_id], 0)
result = ' '.join(result)
print(result)
print("LENGTH: ", len(result.split(' ')))
