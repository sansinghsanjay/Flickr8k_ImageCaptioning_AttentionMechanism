'''
Sanjay Singh
san.singhsanjay@gmail.com
April-2021
scrpt for Image Captioning dataset - Flickr8k
To preprocess captions of training data
'''

# packages
import pandas as pd
import string
from tqdm import tqdm

# function to remove any single character from a string
def removeSingleChar_removeNum(s):
	# removing single characters (such as 'a', 's', etc.) from string s
	tokens = s.split(' ')
	s = ''
	for i in range(len(tokens)):
		if(len(tokens[i]) > 1):
			s = s + tokens[i] + ' '
	s = s.strip()
	# removing words with numbers from string s
	tokens = s.split(' ')
	s = ''
	for i in range(len(tokens)):
		if(tokens[i].isalpha()):
			s = s + tokens[i] + ' '
	s = s.strip()
	return s

# paths
train_file_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/train_image_caption.csv"
target_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/"

# read caption file
train_df = pd.read_csv(train_file_path)

# data cleaning
table = str.maketrans('', '', string.punctuation) # to eliminate all special characters
print("Cleaning data:")
for i in tqdm(range(train_df.shape[0])):
	caption = train_df.iloc[i]['caption']
	caption = caption[1 : len(caption) - 1]
	caption_list = caption.split("<>")
	number_of_captions = len(caption_list) # number of captions for image i
	caption = ''
	for j in range(number_of_captions):
		temp_str = caption_list[j]
		temp_str = temp_str.lower()
		temp_str = temp_str.translate(table)
		temp_str = removeSingleChar_removeNum(temp_str)
		if(len(temp_str) > 5):
			caption = caption + temp_str + "#"
	caption = caption[0 : len(caption) - 1] # removing asterisk added at end
	train_df.iloc[i]['caption'] = caption
print()

# create a vocbulary of all unique words
vocabulary = set() # perfect for making vocabulary - keeps only unique items
vocabulary.update(["<UNK>"])
print("Creating vocabulary:")
for i in tqdm(range(train_df.shape[0])):
	caption = train_df.iloc[i]['caption']
	caption_list = caption.split("#")
	for j in range(len(caption_list)):
		temp_str = caption_list[j]
		vocabulary.update(temp_str.split(' '))
print()

# removing words from captions that are not in vocabulary
max_caption_length = 0
max_i = 0
max_j = 0
print("Removing words from train captions that are not in vocabulary:")
for i in tqdm(range(train_df.shape[0])):
	caption = train_df.iloc[i]['caption']
	caption_list = caption.split("#")
	caption = ''
	for j in range(len(caption_list)):
		temp_str = caption_list[j]
		words = temp_str.split(' ')
		for k in range(len(words)):
			if words[k] not in vocabulary:
				temp_str = temp_str.replace(words[k], '')
				temp_str = temp_str.replace('  ', ' ')
				temp_str = temp_str.strip()
		temp_str = "<startseq> " + temp_str + " <endseq>" # adding startseq and endseq
		if(max_caption_length < len(temp_str.split(" "))):
			max_caption_length = len(temp_str.split(" "))
			max_i = i
			max_j = j
		caption = caption + temp_str + "#"
	caption = caption[0 : len(caption) - 1] # to remove # added at end
	train_df.iloc[i]['caption'] = caption
print()
print("Maximum caption length found (including '<startseq>' and '<endseq>', at index ", max_i, ", ", max_j, "): ", max_caption_length)

# adding startseq and endseq in vocabulary
vocabulary.update(["<startseq>", "<endseq>"])

# saving max_caption_length, max_i and max_j
f_ptr = open(target_path + "max_caption_length.txt", "w")
temp_str = "Max length of caption (including '<startseq>' and '<endseq>', at index " + str(max_i) + ", " + str(max_j) + "): " + str(max_caption_length)
f_ptr.write(temp_str)
f_ptr.close()
print("Saved max length of caption along with index in file max_caption_length.txt")

# saving captions_df
train_df.to_csv(target_path + "train_image_caption_processed.csv", index=False)
print("Saved train_image_caption_processed.csv")

# saving vocabulary into a text file
f_ptr = open(target_path + "vocabulary.txt", "w")
vocabulary = list(vocabulary)
vocabulary.sort()
print("Length of Vocabulary: ", len(vocabulary))
for i in range(len(vocabulary)):
	if(len(vocabulary[i]) > 1):
		f_ptr.write(vocabulary[i] + "\n")
f_ptr.close()
print("Saved vocabulary.txt")
