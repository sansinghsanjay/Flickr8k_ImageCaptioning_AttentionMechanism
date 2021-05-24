'''
Sanjay Singh
san.singhsanjay@gmail.com
May-2021
Training Image Captioning by using Visual Attention Mechanism
Followed this:
https://www.tensorflow.org/tutorials/text/image_captioning
'''

# packages
import tensorflow as tf
import random
import numpy as np
import os
import time
from tqdm import tqdm
import pandas as pd

# constants
NPY_SHAPE_1 = 64
NPY_SHAPE_2 = 2048
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIMS = 256
UNITS = 512
EPOCHS = 20

# paths
images_npy_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_dataset/inception_train_npy_files/"
img_captions_csv_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/train_image_caption_processed.csv"
vocabulary_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/vocabulary.txt"
max_caption_len_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/max_caption_length.txt"
checkpoint_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/trained_model_on_7591files_fullVocab_20Epochs_InceptionV3/"

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

# function to load image name and captions
def load_imagenames_captions(data_path):
	data_df = pd.read_csv(data_path)
	imagenames = list()
	imagecaptions = list()
	print("Loading imagenames and imagecaptions:")
	for i in tqdm(range(data_df.shape[0])):
		caption_value = data_df.iloc[i]['caption']
		captions_list = caption_value.split("#")
		for caption in captions_list:
			imagenames.append(data_df.iloc[i]['image'])
			imagecaptions.append(caption)
	print()
	new_df = pd.DataFrame(columns=['image', 'caption'])
	new_df['image'] = imagenames
	new_df['caption'] = imagecaptions
	# shuffling dataframe
	new_df = new_df.sample(frac=1.0).reset_index(drop=True)
	return new_df

# Load the numpy files
def map_func(img_name, cap):
	img_tensor = np.load(img_name.decode('utf-8')+'.npy')
	return img_tensor, cap

@tf.function
def train_step(img_tensor, target, wordtoix):
	loss = 0
	# initializing the hidden state for each batch
	# because the captions are not related from image to image
	hidden = decoder.reset_state(batch_size=target.shape[0])
	dec_input = tf.expand_dims([wordtoix['<startseq>']] * target.shape[0], 1)
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

# loading data
data_df = load_imagenames_captions(img_captions_csv_path)
print("Data shape: ", data_df.shape)
img_name_train = list(data_df['image'])
cap_train = list(data_df['caption'])

# adding images_npy_path to imagenames
for i in range(len(img_name_train)):
	img_name_train[i] = images_npy_path + img_name_train[i]

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
wordtoix = dict()
for i in range(len(vocabulary)):
	wordtoix[vocabulary[i]] = i
print("Successfully created wordtoix dictionary")

# reading maximum caption length
f_ptr = open(max_caption_len_path, 'r')
data = f_ptr.read()
f_ptr.close()
max_caption_len = int(data.split(":")[1].strip())
print("Max Caption Length: ", max_caption_len)

# creating numerical vector for captions
cap_num_vector = list()
print("Making numerical vector for captions:")
for caption in tqdm(cap_train):
	words = caption.split(' ')
	temp_list = list()
	for word in words:
		temp_list.append(wordtoix[word])
	cap_num_vector.append(temp_list)
print()

# padding zeros in numerical captions to make them all equal to max_caption_len
cap_num_vector = tf.keras.preprocessing.sequence.pad_sequences(cap_num_vector, max_caption_len, padding='post')

# making dataset
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_num_vector))
dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=tf.data.AUTOTUNE)
# shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# optimizer and loss function
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# making object of encoder and decoder
encoder = CNN_Encoder(EMBEDDING_DIMS)
decoder = RNN_Decoder(EMBEDDING_DIMS, UNITS, vocab_size)

# creating checkpoint for model
ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
	start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
	# restoring the latest checkpoint in checkpoint_path
	ckpt.restore(ckpt_manager.latest_checkpoint)

# to collect loss values
loss_values = list()

# training network
start_epoch = 0
num_steps = len(img_name_train) // BATCH_SIZE
for epoch in range(start_epoch, EPOCHS):
	start = time.time()
	total_loss = 0
	for (batch, (img_tensor, target)) in enumerate(dataset):
		batch_loss, t_loss = train_step(img_tensor, target, wordtoix)
		total_loss += t_loss
		if batch % 100 == 0:
			average_batch_loss = batch_loss.numpy()/int(target.shape[1])
			print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')
	# storing the epoch end loss value to plot later
	loss_values.append(total_loss / num_steps)
	print(f'Epoch {epoch+1} Loss {total_loss/num_steps:.6f}')
	print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')

# saving model
ckpt_manager.save()
