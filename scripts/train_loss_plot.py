'''
Sanjay Singh
san.singhsanjay@gmail.com
May-2021
To create plot of loss values generated while training
'''

# packages
import matplotlib.pyplot as plt

# paths
train_log_file_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/trained_model/train_log.txt"
target_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning_AttentionMechanism/output/trained_model/"

# reading train log
f_ptr = open(train_log_file_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
loss_values = list() 
for line in lines: 
	line = line.strip() 
	tokens = line.split(' ') 
	if("Batch" not in tokens and "sec" not in tokens and len(tokens) > 1): 
		loss_values.append(float(tokens[len(tokens) - 1]))
last_line = lines[len(lines) - 1]
last_line = last_line.strip()
tokens = last_line.split(' ')
loss_values.append(float(tokens[len(tokens) - 1]))

# epochs
epochs = len(loss_values)
epoch_values = list(range(1, epochs + 1))

# making plot
plt.scatter(epoch_values, loss_values)
plt.plot(epoch_values, loss_values)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.xticks(epoch_values)
plt.ylabel("Loss Value")
plt.grid()
plt.savefig(target_path + "train_loss_plot.png")
