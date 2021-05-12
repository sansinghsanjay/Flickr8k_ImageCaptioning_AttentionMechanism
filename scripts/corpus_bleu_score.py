'''
Sanjay Singh
san.singhsanjay@gmail.com
May-2021
This script is to evalute the result of Image Captioning model on the entire test dataset.
'''

# packages
import pandas as pd
import nltk.translate.bleu_score as bleu

# paths
candidate_captions_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/output/generated_captions/val_captions.txt" # generate captions
references_captions_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/output/intermediate_files/val_image_caption_processed.csv" # actual captions

# read reference captions csv file
ref_df = pd.read_csv(references_captions_path)

# converting ref_df to a dictionary
ref_dict = dict()
for i in range(ref_df.shape[0]):
	ref_dict[ref_df.iloc[i]['image']] = ref_df.iloc[i]['caption']

# read candidate captions txt file
cand_dict = dict()
f_ptr = open(candidate_captions_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
image_name = list()
caption_list = list()
for line in lines:
	splits = line.split("#")
	image_name.append(splits[0])
	caption_list.append(splits[1])

# putting all candidate values in a dictionary
for i in range(len(image_name)):
	cand_dict[image_name[i]] = caption_list[i]

# making list of candidate and reference captions
ref_list = list()
cand_list = list()
for img_name, caption_text in cand_dict.items():
	cand_list.append(caption_text.split())
	captions = ref_dict[img_name]
	captions_list = captions.split("#")
	temp_list = list()
	for i in range(len(captions_list)):
		splits = captions_list[i].split()
		splits = splits[1 : len(splits) - 1]
		temp_list.append(splits)
	ref_list.append(temp_list)

# get bleu score
bleu_sc = bleu.corpus_bleu(ref_list, cand_list)
print("Corpus BLEU Score: ", bleu_sc)
