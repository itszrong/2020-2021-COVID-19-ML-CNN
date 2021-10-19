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

classes = ('Covid', 'nonCovid')

def load_images_from_folder(Covid, nonCovid):
	myDataset = []
	for filename in os.listdir(Covid):
		if filename == ".DS_Store":
			continue
		img = Image.open(os.path.join(Covid, filename)).convert('L')
		rsize = img.resize((227,227)) #so that all images are the same
		# display(rsize)
		rsize = np.asarray(rsize)
		rsize = (rsize - np.mean(rsize)) / np.std(rsize)
		# move channels to first dimension and add to stack
		rsize = np.transpose(rsize)
		myDataset.append((rsize, np.asarray(1)))
	for filename in os.listdir(nonCovid):
		if filename == ".DS_Store":
			continue
		img = Image.open(os.path.join(nonCovid, filename)).convert('L') #so that all images are the same
		# display(rsize)
		rsize = img.resize((227,227))
		rsize = np.asarray(rsize)
		rsize = (rsize - np.mean(rsize)) / np.std(rsize)
		# move channels to first dimension and add to stack
		rsize = np.transpose(rsize)
		myDataset.append((rsize, np.asarray(0)))
	return myDataset

print("Hello world 1")
Covid, nonCovid = ("CT_COVID", "CT_NonCOVID")
dataset = load_images_from_folder(Covid, nonCovid)

images = np.zeros((len(dataset), 1, 227, 227))
labels = np.zeros(len(dataset)).astype(int)

for i in range(len(dataset)):
	images[i] = dataset[i][0]
	labels[i] = dataset[i][1]
print("Hello world x")

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

#random.shuffle(dataset)

print("Hello world 2")

direct_dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_imgs), torch.LongTensor(train_labels))
direct_dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_imgs), torch.LongTensor(val_labels))
# direct_dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_imgs), torch.LongTensor(val_labels), torch.Tensor(indices))


batchsize = 4
trainloader = torch.utils.data.DataLoader(direct_dataset_train, batch_size=batchsize, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(direct_dataset_val, batch_size=batchsize, shuffle=True, num_workers=0)

print("Hello world 3")

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
		self.fc3 = nn.Linear(4096, 1)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.pool(F.relu(self.conv5(x)))
		x = x.view(-1, 256*6*6)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = torch.sigmoid(self.fc3(x))
		return x

print("Hello world 4")

#part of object orientated programming
net = Net()
print(net.parameters())
# model = models.vgg16()

criterion = nn.BCELoss() #BCELoss change array to 1D because binary
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9) #holds current parameters and updates accordingly
print("Hello world 5")


best_validation_loss = np.Inf
validation_loss_over_time = []
for epoch in range(3):
	running_loss = 0.0
	validation_loss = 0.0
	net.train(True)
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		# len(trainloader) = 149
		optimizer.zero_grad() # zeros the parameter gradients
		#forward + backward + optimise
		outputs = net(inputs)
		loss = criterion(outputs, labels.unsqueeze(1).float())
		loss.backward()
		optimizer.step()
		#print statistics
		running_loss += loss.item()
		if i % 50 == 49:
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/(i+1))) #divide by i instead of 50
			# running_loss = 0.0
	#validation loop
	# correct = 0
	# total = 0
	with torch.no_grad():
		for data in valloader:
			images, labels = data
			outputs = net(images)
			loss = criterion(outputs, labels.unsqueeze(1).float())
			validation_loss += loss.item()
			# _, predicted = torch.max(outputs.data, 1) #CNL includes sigmoid, torch.argmax?
			# total += labels.size(0)
			# correct += (predicted == labels).sum().item()
	print("Validation loss is", validation_loss/len(valloader))
	validation_loss_over_time.append(validation_loss/len(valloader))
	if validation_loss/len(valloader) < best_validation_loss:
		torch.save(net.state_dict(), 'net.pth')
		best_validation_loss = validation_loss/len(valloader)
print("Hello world 10")

tensor_val_imgs = torch.Tensor(val_imgs)
tensor_val_labels = torch.LongTensor(val_labels)
print('tensor_val_labels', tensor_val_labels)
outputs_val = net(tensor_val_imgs)
print("Hello world 11")

_, predicted = torch.max(outputs_val, 1)
print('Actual: ', ' '.join('%5s' % classes[tensor_val_labels[j]] for j in range(4)))
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

print("Hello world 12")
correct = 0
total = 0
with torch.no_grad(): # running a test instead of training the learner
	for data in valloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)

		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print("Total number of images for test is ", total)
print("Total number of correct predictions for test is ", correct)
print('Accuracy of the network on the', total,'test images: %d %%' % (100 * correct / total))

















