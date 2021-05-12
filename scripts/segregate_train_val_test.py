'''
Sanjay Singh
san.singhsanjay@gmail.com
April-2021
To segregate train, validation and test data
'''

# packages
import pandas as pd

# function to update progress
def percentage_progress(completed, total):
	perc_progress = (completed / total) * 100
	perc_progress = round(perc_progress, 2)
	return perc_progress

# paths
entire_data_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_dataset/captions.txt"
train_data_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/archive/Flickr_8k.trainImages.txt"
val_data_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/archive/Flickr_8k.valImages.txt"
test_data_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/archive/Flickr_8k.testImages.txt"
target_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/output/intermediate_files/"

# reading entire data
f_name = ""
df_loc = 0
image_name = list()
captions = ''
entire_df = pd.DataFrame(columns=['image', 'caption'])
f_ptr = open(entire_data_path, 'r')
lines = f_ptr.readlines()
for i in range(len(lines)):
	if(i == 0): # header line, thus ignore it
		continue
	else:
		line = lines[i].strip()
		temp_str = line.split(".")[0] + ".jpg"
		cap = ' '.join(line.split(".")[1:])
		cap = cap[3:]
		if(f_name != temp_str and i == 1):
			f_name = temp_str
			captions = cap
		if(f_name == temp_str and i > 1):
			captions = captions + '<>' + cap
		if(f_name != temp_str and i > 1):
			entire_df.loc[df_loc] = [f_name, captions]
			df_loc += 1
			f_name = temp_str
			captions = cap
	if((i + 1) % 500 == 0 or (i + 1) == len(lines)):
		perc_progress = percentage_progress((i + 1), len(lines))
		print("Completed grouping image name and captions: ", perc_progress, " %", end='\r')
print()
entire_df.loc[df_loc] = [f_name, captions]

# reading train data
train_filenames = list()
f_ptr = open(train_data_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
for line in lines:
	train_filenames.append(line.strip())

# reading val data
val_filenames = list()
f_ptr = open(val_data_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
for line in lines:
	val_filenames.append(line.strip())

# reading test data
test_filenames = list()
f_ptr = open(test_data_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
for line in lines:
	test_filenames.append(line.strip())

# grouping captions in entire data based on image name
#entire_df = entire_df.groupby(['image'])
#entire_df = entire_df['caption'].apply(list)
#entire_df = entire_df.reset_index()

# extracting image-caption for training data
train_df = pd.DataFrame()
for i in range(len(train_filenames)):
	for j in range(entire_df.shape[0]):
		if(train_filenames[i] == entire_df.iloc[j]['image']):
			train_df = train_df.append(entire_df.iloc[j])
			entire_df = entire_df.drop(entire_df.index[j]).reset_index(drop=True) # drop this row as we have extracted this, this will reduce size and fast further computations
			break
	perc_progress = percentage_progress((i + 1), len(train_filenames))
	print("Completed creating train - image and caption dataframe: ", perc_progress, " %", end='\r')
train_df = train_df.reset_index(drop=True)
train_df = train_df[['image', 'caption']] # changing sequence of cols

# saving train_df
train_df.to_csv(target_path + "train_image_caption.csv", index=False)
print("Saved train data - image and captions")

# extracting image-caption for validation data
val_df = pd.DataFrame()
for i in range(len(val_filenames)):
	for j in range(entire_df.shape[0]):
		if(val_filenames[i] == entire_df.iloc[j]['image']):
			val_df = val_df.append(entire_df.iloc[j])
			entire_df = entire_df.drop(entire_df.index[j]).reset_index(drop=True) # drop this row as we have extracted this, this will reduce size and fast further computations
			break
	perc_progress = percentage_progress((i + 1), len(val_filenames))
	print("Completed creating val - image and caption dataframe: ", perc_progress, " %", end='\r')
val_df = val_df.reset_index(drop=True)
val_df = val_df[['image', 'caption']] # changing sequence of cols

# saving val_df
val_df.to_csv(target_path + "val_image_caption.csv", index=False)
print("Saved validation data - image and captions")

# extracting image-caption for test data
test_df = pd.DataFrame()
for i in range(len(test_filenames)):
	for j in range(entire_df.shape[0]):
		if(test_filenames[i] == entire_df.iloc[j]['image']):
			test_df = test_df.append(entire_df.iloc[j])
			entire_df = entire_df.drop(entire_df.index[j]).reset_index(drop=True) # drop this row as we have extracted this, this will reduce size and fast further computations
			break
	perc_progress = percentage_progress((i + 1), len(test_filenames))
	print("Completed creating test - image and caption dataframe: ", perc_progress, " %", end='\r')
test_df = test_df.reset_index(drop=True)
test_df = test_df[['image', 'caption']] # changing sequence of cols

# saving test_df
test_df.to_csv(target_path + "test_image_caption.csv", index=False)
print("Saved test data - image and captions")

