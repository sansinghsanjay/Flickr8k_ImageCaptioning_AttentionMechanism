'''
Sanjay Singh
san.singhsanjay@gmail.com
April-2021
To segregate train, validation and test data
'''

# packages
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# paths
entire_data_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/archive/captions.txt"
train_data_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/archive/Flickr_8k.trainImages.txt"
val_data_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/archive/Flickr_8k.valImages.txt"
test_data_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/archive/Flickr_8k.testImages.txt"
target_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/intermediate_files/"

# reading entire data
f_name = ""
df_loc = 0
image_name = list()
captions = ''
entire_df = pd.DataFrame(columns=['image', 'caption'])
f_ptr = open(entire_data_path, 'r')
lines = f_ptr.readlines()
print("Reading entire data and grouping captions based on image name:")
for i in tqdm(range(len(lines))):
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

# converting entire_df from dataframe to dictionary for fast access
entire_dict = entire_df.set_index('image').T.to_dict('list')

# extracting image-caption for training data
# since model performed poor on training data, thus we need to increase data
# thus, adding validation data and half of testing data for training
train_df = pd.DataFrame(columns=['image', 'caption'])
print("Collecting data for training from the provided training set:")
for i in tqdm(range(len(train_filenames))):
	row = {'image': train_filenames[i], 'caption': entire_dict[train_filenames[i]]}
	train_df = train_df.append(row, ignore_index=True)
print()
print("Shape of training dataframe after collecting data from train set: ", train_df.shape)
print("Collecting data for training from the provided validation set:")
for i in tqdm(range(len(val_filenames))):
	row = {'image': val_filenames[i], 'caption': entire_dict[val_filenames[i]]}
	train_df = train_df.append(row, ignore_index=True)
print()
print("Shape of training dataframe after collecting data from validation set: ", train_df.shape)
print("Collecting half of test data for training: ")
half_test_data = int(len(test_filenames) / 2)
for i in tqdm(range(0, half_test_data)):
	row = {'image': test_filenames[i], 'caption': entire_dict[test_filenames[i]]}
	train_df = train_df.append(row, ignore_index=True)
print()
print("Shape of training dataframe after collecting half of test data: ", train_df.shape)
test_filenames = test_filenames[half_test_data:]
print("Number of files in test data after taking half of them in training: ", len(test_filenames))

# saving train_df
train_df.to_csv(target_path + "train_image_caption.csv", index=False)
print("Saved train data - image and captions")

# extracting test data from the provided test set
test_df = pd.DataFrame(columns=['image', 'caption'])
print("Collecting test data from the provided test data:")
for i in tqdm(range(len(test_filenames))):
	row = {'image': test_filenames[i], 'caption': entire_dict[test_filenames[i]]}
	test_df = test_df.append(row, ignore_index=True)
print()
print("Shape of test data: ", test_df.shape)

# saving test_df
test_df.to_csv(target_path + "test_image_caption.csv", index=False)
print("Saved test data - image and captions")

# making plot for number of files in original, training and testing data
labels = ['Total', 'Train', 'Test']
values = [entire_df.shape[0], train_df.shape[0], test_df.shape[0]]
fig, ax = plt.subplots()
plt.title("Number of Images in Total, Train and Test Dataset")
plt.xlabel("Name of dataset")
plt.ylabel("Number of images")
plt.grid()
plt.bar(labels, values, color='blue')
for index, value in enumerate(values):
	ax.text(index, value, str(value), color='black')
plt.savefig(target_path + "no_of_imgs_in_total_train_test.png")
print("Plot for number of images in total, training and test is saved successfully")
