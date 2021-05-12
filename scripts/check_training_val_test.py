'''
Sanjay Singh
san.singhsanjay@gmail.com
April-2021
To check training, validation and test data set
'''

# packages
import matplotlib.pyplot as plt

# paths
original_data_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/archive/captions.txt" 
train_data_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/archive/Flickr_8k.trainImages.txt"
validation_data_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/archive/Flickr_8k.valImages.txt"
test_data_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/archive/Flickr_8k.testImages.txt"
target_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/archive/" 

# read original data
original_filenames = set()
f_ptr = open(original_data_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
for line in lines:
	line = line.split(",")[0]
	line = line.strip()
	original_filenames.add(line)
original_filenames = list(original_filenames)

# status of original data
print("Number of images in original data: ", len(original_filenames))

# read training data
train_filenames = list()
f_ptr = open(train_data_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
for line in lines:
	train_filenames.append(line.strip())
print("Completed reading training filenames")

# read test data
test_filenames = list()
f_ptr = open(test_data_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
for line in lines:
	test_filenames.append(line.strip())
print("Completed reading test filenames")

# updating status
print("Number of images in train data: ", len(train_filenames))
print("Number of images in test data: ", len(test_filenames))

# check if any image hasn't repeated in training and testing data
duplicate_found = False
for i in range(len(test_filenames)):
	if(test_filenames[i] in train_filenames):
		print("Duplication found at index ", i, ", name: ", test_filenames[i])
		duplicate_found = True
if(duplicate_found == False):
	print("No duplicates found in training and testing data")

# check whether all train and test images are available in original data or not
for i in range(len(train_filenames)):
	if(train_filenames[i] not in original_filenames):
		print("Train image not found, at index: ", i, ", name: ", train_filenames[i])
for i in range(len(test_filenames)):
	if(test_filenames[i] not in original_filenames):
		print("Test image not found, at index: ", i, ", name: ", test_filenames[i])
print("Completed checking if train and test filenames are in original dataset or not")

# extracting filenames for validation dataset
validation_filenames = list()
f_ptr = open(validation_data_path, "w")
for i in range(len(original_filenames)):
	if(original_filenames[i] not in train_filenames and original_filenames[i] not in test_filenames):
		validation_filenames.append(original_filenames[i])
		f_ptr.write(original_filenames[i] + '\n')
f_ptr.close()
print("Completed writing validation filenames")

# status of validation filenames
print("Number of validation filenames: ", len(validation_filenames))

# making plot for number of files in original, training, validation and testing data
labels = ['Original', 'Training', 'Validation', 'Testing']
values = [len(original_filenames), len(train_filenames), len(validation_filenames), len(test_filenames)]
fig, ax = plt.subplots()
plt.title("Number of images in original, train, validation and test dataset")
plt.xlabel("Name of dataset")
plt.ylabel("Number of images")
plt.grid()
plt.bar(labels, values, color='blue')
for index, value in enumerate(values):
	ax.text(index, value, str(value), color='black')
plt.savefig(target_path + "no_of_imgs_in_original_train_val_test.png")
print("Plot for number of images in original, training, validation and test is saved successfully")
