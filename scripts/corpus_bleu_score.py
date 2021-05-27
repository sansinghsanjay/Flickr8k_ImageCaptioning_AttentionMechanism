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
candidate_captions_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/generated_captions/test_data_predicted_captions.csv" # generate captions
references_captions_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/test_image_caption_processed.csv" # actual captions

# read reference captions csv file
ref_df = pd.read_csv(references_captions_path)

# converting ref_df to a dictionary
ref_dict = dict()
for i in range(ref_df.shape[0]):
	ref_dict[ref_df.iloc[i]['image']] = ref_df.iloc[i]['caption']

# read candidate captions csv file
cand_df = pd.read_csv(candidate_captions_path)

# putting all candidate values in a dictionary
cand_dict = dict()
for i in range(cand_df.shape[0]):
	imagename = cand_df.iloc[i]['image'].split("/")
	imagename = imagename[len(imagename) - 1]
	cand_dict[imagename] = cand_df.iloc[i]['predicted captions']

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
