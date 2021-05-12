'''
Sanjay Singh
san.singhsanjay@gmail.com
April-2021
To create a subset of data - extracting a few image names and their captions for testing attention mechanism code (quick test)
'''

import pandas as pd

data_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/output/intermediate_files/train_image_caption_processed.csv"

data = pd.read_csv(data_path)

print("Shape: ", data.shape)
print("HEAD: ")
print(data.head())
print("TAIL: ")
print(data.tail())

img_name = list()
img_caption = list()

for i in range(data.shape[0]):
	temp_str = data.iloc[i]['image']
	captions = data.iloc[i]['caption']
	captions = captions.split("#")
	for cap in captions:
		img_name.append(temp_str)
		img_caption.append(cap)
	if((i + 1) == 10):
		break

df = pd.DataFrame(columns=['image', 'caption'])
df['image'] = img_name
df['caption'] = img_caption
df.to_csv("/home/sansingh/github_repo/Flickr8k_ImageCaptioning/output/intermediate_files/mini_data.csv", index=False)
