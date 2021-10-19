from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from IPython.display import display
import random
import math

classes = ('Covid', 'nonCovid')

def load_images_from_folder(Covid, nonCovid):
	myDataset = []
	myImages = []
	for filename in os.listdir(Covid):
		if filename == ".DS_Store":
			continue
		img = Image.open(os.path.join(Covid, filename))
		myImages.append(np.asarray(img))
		img = img.convert('L')
		rsize = img.resize((227,227)) #so that all images are the same
		# display(rsize)
		rsize = np.asarray(rsize)
		rsize = (rsize - np.mean(rsize)) / np.std(rsize)
		# move channels to first dimension and add to stack
		myDataset.append((rsize, np.asarray(1)))
	for filename in os.listdir(nonCovid):
		if filename == ".DS_Store":
			continue
		img = Image.open(os.path.join(nonCovid, filename)) #so that all images are the same
		myImages.append(np.asarray(img))
		img = img.convert('L')
		# display(rsize)
		rsize = img.resize((227,227))
		rsize = np.asarray(rsize)
		rsize = (rsize - np.mean(rsize)) / np.std(rsize)
		# move channels to first dimension and add to stack
		myDataset.append((rsize, np.asarray(0)))
	return myDataset, myImages

print("1: Initialise")
Covid, nonCovid = ("CT_COVID", "CT_NonCOVID")
dataset, myImages = load_images_from_folder(Covid, nonCovid)

images = np.zeros((len(dataset), 1, 227, 227))
labels = np.zeros(len(dataset)).astype(int)

for i in range(len(dataset)):
	images[i] = dataset[i][0]
	labels[i] = dataset[i][1]

print("The dataset has", len(dataset), "images.")

indices = torch.randperm(len(dataset)).tolist()

split = int(round(len(indices)*0.8))
train_imgs = []
train_labels = []
val_imgs = []
val_labels = []

train_imgs = images[indices[:split]]
train_labels = labels[indices[:split]]
val_imgs = images[indices[split+1:]]
val_labels = labels[indices[split+1:]]

train_imgs_unaltered = []
val_imgs_unaltered = []
for i in range(len(myImages)):
	if i < split:
		train_imgs_unaltered.append(myImages[indices[i]])
	else:
		val_imgs_unaltered.append(myImages[indices[i]])

#display images - preliminary check
no_images_displayed = 5
_, axarr = plt.subplots(1,no_images_displayed)
for i in range(no_images_displayed):
	axarr[i].imshow(val_imgs_unaltered[i])
	axarr[i].set_title(classes[val_labels[i]])
plt.show()

print("2: Data loaded")
direct_dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_imgs), torch.LongTensor(train_labels))
direct_dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_imgs), torch.LongTensor(val_labels))

batchsize = 4
trainloader = torch.utils.data.DataLoader(direct_dataset_train, batch_size=batchsize, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(direct_dataset_val, batch_size=batchsize, shuffle=True, num_workers=0)

print("3: Shuffled and batched")

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 96, kernel_size = 11, stride = 4)
		self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2)
		self.conv2 = nn.Conv2d(96, 256, kernel_size = 5, padding = 2)
		self.conv3 = nn.Conv2d(256, 384,kernel_size = 3, padding = 1)
		self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, padding = 1)
		self.conv5 = nn.Conv2d(384, 256, kernel_size = 3, padding = 1)
		self.fc1 = nn.Linear(256 * 6 * 6, 4096)
		self.fc2 = nn.Linear(4096, 4096)
		self.fc3 = nn.Linear(4096, 2)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.pool(F.relu(self.conv5(x)))
		x = x.view(-1, 256*6*6)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

def fn_valloader(valloader):
	validation_loss_total= 0.0
	with torch.no_grad():
		for data in valloader:
			images, labels = data
			outputs = net(images)
			loss = criterion(outputs, labels)
			validation_loss_total += loss.item()
			# _, predicted = torch.max(outputs.data, 1) #CNL includes sigmoid, torch.argmax?
			# total += labels.size(0)
			# correct += (predicted == labels).sum().item()
	return validation_loss_total/len(valloader)

def fn_running_loss(trainloader):
	running_loss_total = 0.0
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		inputs.requires_grad
		optimizer.zero_grad() # zeros the parameter gradients
		#forward + backward + optimise
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		#print statistics
		running_loss_total += loss.item()
		running_loss = running_loss_total/(i+1)
		if i % math.floor(len(trainloader)/3) == math.floor(len(trainloader)/3)-1:
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, (running_loss_total/(i+1))))
	return running_loss


def fn_accuracy(valloader):
	correct = 0
	total = 0
	with torch.no_grad():  # running a test instead of training the learner
		for data in valloader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)  # returns value and indices, so prediction in the 2nd part
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	accuracy = (100 * correct / total)
	print("Total number of images for test is ", total)
	print("Total number of correct predictions for test is ", correct)
	print('Accuracy of the network on the', total, 'test images: %d %%' % accuracy)
	return accuracy


print("4: Model defined")

#part of object orientated programming
net = Net()
# print(net.parameters())
# model = models.vgg16()

criterion = nn.CrossEntropyLoss() #BCELoss change array to 1D because binary
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9) #holds current parameters and updates accordingly
print("5: Optimiser and criterion defined")
print("6: Training begins")


best_validation_loss = np.Inf
running_loss_over_epochs = []
validation_loss_over_epochs = []
accuracy_over_epochs = []
epochs = 2

for epoch in range(epochs):
	net.train(True)

	running_loss = fn_running_loss(trainloader)
	running_loss_over_epochs.append(running_loss)

	validation_loss = fn_valloader(valloader)
	print("Validation loss is", validation_loss)

	validation_loss_over_epochs.append(validation_loss)
	if validation_loss < best_validation_loss:
		torch.save(net.state_dict(), 'net.pth')
		best_validation_loss = validation_loss

	accuracy = fn_accuracy(valloader)
	accuracy_over_epochs.append(accuracy)

print("7: Training complete")

epochs_list = list(range(1,epochs+1))
plot1 = plt.figure(1)
plt.plot(epochs_list, running_loss_over_epochs, 'r', label="Running loss")
plt.plot(epochs_list, validation_loss_over_epochs, 'b', label="Validation loss")
plt.legend(loc='upper right')
plt.title('Loss against Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plot2 = plt.figure(2)
plt.plot(epochs_list, accuracy_over_epochs, 'k', label = "accuracy")
plt.legend(loc='upper right')
plt.title('Accuracy against Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

print("8: Learning curves and accuracy plotted against epochs")

tensor_val_imgs = torch.Tensor(val_imgs)
tensor_val_labels = torch.LongTensor(val_labels)
# print('tensor_val_labels', tensor_val_labels)
outputs_val = net(tensor_val_imgs)
print("9: Initialised tensors for midway prediction test")

_, predicted = torch.max(outputs_val, 1)
print('Actual: ', ' '.join('%5s' % classes[tensor_val_labels[j]] for j in range(4)))
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


#display images
no_images_displayed = 10
_, axarr = plt.subplots(1,no_images_displayed)
for i in range(no_images_displayed):
	axarr[i].imshow(val_imgs_unaltered[i])
	axarr[i].set_title('Ground truth:' + classes[val_labels[i]] +'\nPredicted:' + classes[predicted[i]])
plt.show()
print("10: Images plotted for visual check")

print("11: Accuracy on training set")
correct = 0
total = 0
with torch.no_grad(): # running a test instead of training the learner
	for data in trainloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1) #returns value and indices, so prediction in the 2nd part
		# print(predicted)
		# print(predicted.shape)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()


print("Total number of images for test is ", total)
print("Total number of correct predictions for test is ", correct)
print('Accuracy of the network on the', total,'train images: %d %%' % (100 * correct / total))

print("12: Accuracy on validation set")
correct = 0
total = 0
with torch.no_grad(): # running a test instead of training the learner
	for data in valloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1) #returns value and indices, so prediction in the 2nd part
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print("Total number of images for test is ", total)
print("Total number of correct predictions for test is ", correct)
print('Accuracy of the network on the', total,'test images: %d %%' % (100 * correct / total))

print("13: Run complete")
















