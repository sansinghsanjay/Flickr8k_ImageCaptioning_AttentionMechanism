'''
Sanjay Singh
san.singhsanjay@gmail.com
April-2021
scrpt for Image Captioning dataset - Flickr8k
'''

# packages
import pandas as pd
import string

# GLOBAL CONST
WORD_COUNT_THRESH = 10

# paths
val_file_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/val_image_caption.csv"
test_file_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/test_image_caption.csv"
vocabulary_file_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/vocabulary.txt"
target_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/"

# fucntion for finding progress in percentage
def percentage_progress(completed, total):
	perc_progress = (completed / total) * 100
	perc_progress = round(perc_progress, 2)
	return perc_progress

# function to do data cleaning
def data_cleaning(df):
	table = str.maketrans('', '', string.punctuation) # to eliminate all special characters
	for i in range(df.shape[0]):
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
		if((i + 1) % 500 == 0 or (i + 1) == df.shape[0]): # update status
			perc_progress = percentage_progress((i + 1), df.shape[0])
			print("Completed pre-processing of train captions: ", perc_progress, " %", end='\r')
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
def remove_words_not_in_vocabulary(df):
	for i in range(df.shape[0]):
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
		if((i + 1) % 1000 == 0 or (i + 1) == df.shape[0]):
			perc_progress = percentage_progress((i + 1), df.shape[0]) 
			print("Completed removing words from captions that are not in vocabulary: ", perc_progress, " %", end='\r')
	print()
	return df


# read caption file
val_df = pd.read_csv(val_file_path)
test_df = pd.read_csv(test_file_path)

# data cleaning
val_df = data_cleaning(val_df)
test_df = data_cleaning(test_df)

# reading vocabulary file
f_ptr = open(vocabulary_file_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
vocabulary = set()
for line in lines:
	vocabulary.update(line.strip())

# removing words from captions that are not in vocabulary
val_df = remove_words_not_in_vocabulary(val_df)
test_df = remove_words_not_in_vocabulary(test_df)

# saving captions_df
val_df.to_csv(target_path + "val_image_caption_processed.csv", index=False)
test_df.to_csv(target_path + "test_image_caption_processed.csv", index=False)
print("Saved val_image_caption_processed.csv and test_image_caption_processed.csv successfully")
