'''
Sanjay Singh
san.singhsanjay@gmail.com
April-2021
scrpt for Image Captioning dataset - Flickr8k
'''

# packages
import pandas as pd
import string
from tqdm import tqdm

# GLOBAL CONST
WORD_COUNT_THRESH = 10

# paths
test_file_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/test_image_caption.csv"
vocabulary_file_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/vocabulary.txt"
target_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/"

# function to do data cleaning
def data_cleaning(df):
	table = str.maketrans('', '', string.punctuation) # to eliminate all special characters
	for i in tqdm(range(df.shape[0])):
		caption = df.iloc[i]['caption']
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
		df.iloc[i]['caption'] = caption
	print()
	return df

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

# removing words from captions that are not in vocabulary
def remove_words_not_in_vocabulary(df, vocabulary):
	for i in tqdm(range(df.shape[0])):
		caption = df.iloc[i]['caption']
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
			caption = caption + temp_str + "#"
		caption = caption[0 : len(caption) - 1] # to remove # added at end
		df.iloc[i]['caption'] = caption
	print()
	return df

# read caption file
test_df = pd.read_csv(test_file_path)

# data cleaning
test_df = data_cleaning(test_df)

# reading vocabulary file
f_ptr = open(vocabulary_file_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
vocabulary = list()
for line in lines:
	vocabulary.append(line.strip())

# removing words from captions that are not in vocabulary
test_df = remove_words_not_in_vocabulary(test_df, vocabulary)

# saving captions_df
test_df.to_csv(target_path + "test_image_caption_processed.csv", index=False)
print("Saved test_image_caption_processed.csv successfully")
