'''
Sanjay Singh
san.singhsanjay@gmail.com
May-2021
script for processing (or cleaning) captions (text data) - Image Captioning
'''

# packages
import pandas as pd
import string

# GLOBAL CONST
WORD_COUNT_THRESH = 10

# fucntion for finding progress in percentage
def percentage_progress(completed, total):
	perc_progress = (completed / total) * 100
	perc_progress = round(perc_progress, 2)
	return perc_progress

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
file_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/output/intermediate_files/val_image_caption.csv"
vocabulary_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/output/intermediate_files/vocabulary.txt"
target_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/output/intermediate_files/"

# read vocabulary file
vocabulary = list()
f_ptr = open(vocabulary_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
for line in lines:
	vocabulary.append(line.strip())
print("Successfully loaded vocabulary")

# read caption file
df = pd.read_csv(file_path)

# data cleaning
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
		print("Completed pre-processing of captions: ", perc_progress, " %", end='\r')
print()

# removing words from captions that are not in vocabulary
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
		temp_str = "startseq " + temp_str + " endseq" # adding startseq and endseq
		caption = caption + temp_str + "#"
	caption = caption[0 : len(caption) - 1] # to remove # added at end
	df.iloc[i]['caption'] = caption
	if((i + 1) % 1000 == 0 or (i + 1) == df.shape[0]):
		perc_progress = percentage_progress((i + 1), df.shape[0]) 
		print("Completed removing words from captions that are not in vocabulary: ", perc_progress, " %", end='\r')
print()

# saving captions_df
df.to_csv(target_path + "val_image_caption_processed.csv", index=False)
print("Saved successfully")
