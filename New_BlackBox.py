#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 22:10:54 2020

reference paper : Practical Black-Box Attacks against Machine Learning
arXiv:1602.02697v4 [cs.CR] 19 Mar 2017

@author: ihpark
"""

from __future__ import print_function ## ??
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.autograd import Variable


# parameters of the network
netB_train_num = 150 # the number of training data for the F network
epsilon = 0.3 # data sensitivity for adversarial attack
lamb = 0.1 # data augumentation step size
num = netB_train_num + 1 # New dictionary's key count
iter_epoch = 10
sub_epoch = 6
epoch_num = iter_epoch*sub_epoch # total epoch. the count will be a (substitute epoch*iteration epoch). 
tau = 1 # lamda change period
train_flag = 0 # if you want to train the F network, set the value 1
PATH_Result = '/home/ihpark/Adversarial Attack/Result_Black'

# After sigma epoch, the number of augmentation data will reduce through k_num
sigma = 2
sigma = sigma +1
k_num = 400 


# Reset model's parameters after a subsitute epoch
# If you want to know about subsitute epoch, need to see a related paper
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
        # isinstance(m,nn.Conv2d) : whether instance exist 
    

# FGSM attack 
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# After the sigma epoch, the number of increasing augmentation data will be decreased. 
# This function reflect the k_num, lou, sigma. 
def sampling_func(lou, sigma,  k_num, Dict_len):
    N = int(round(k_num*(lou-sigma)))
    A_list = list(range(1,Dict_len+1))
    index = np.random.choice(A_list, N,replace = 0)
    # return is array
    return index

# Reference network called 'oracle' in the black box adversarial attack paper
class Net_A(nn.Module):
    def __init__(self):
        super(Net_A, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# This is a F network which  trains the oracle's output O(x')
class Net_B(nn.Module):
    def __init__(self):
        super(Net_B, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding = 1)
        self.fc1 = nn.Linear(7*7*64,200)
        self.fc2 = nn.Linear(200,10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


# Path where you want to download MNIST data.    
PATH_Oracle = '../data/MNIST_net.pth'


use_cuda=True

# training loader from MNIST data :  it contains two type of data, one is about image data and  another is image labels.
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=16, shuffle=True)



net_A = Net_A()
net_B = Net_B()
criterion = nn.CrossEntropyLoss()
optimizer_net_A = optim.SGD(net_A.parameters(), lr=0.001, momentum = 0.9)
optimizer_net_B = optim.SGD(net_B.parameters(), lr=0.001, momentum = 0.9)

print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
model_A = net_A.to(device)
model_B = net_B.to(device)



# Oracle train
if not os.path.isfile(PATH_Oracle) :
    for epoch in range(50):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels= inputs.to(device), labels.to(device)
            optimizer_net_A.zero_grad()
            outputs = net_A(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_net_A.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training base network')
    torch.save(net_A.state_dict(),PATH_Oracle)


print('load pretrained base network_A data ')

# The model_A must be in evaluation mode.
model_A.load_state_dict(torch.load(PATH_Oracle, map_location='cpu'))
model_A.eval()

# This code set two Dictionary file. there are data, label Dictionaries to train the network F.
Data_Dict = {}
Label_Dict = {}


#  Test_loader for learning the network F.
test_loader = torch.utils.data.DataLoader(
  datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
          transforms.ToTensor(),
          ])),
      batch_size=1, shuffle=False)

# For augmentation, the data is devided to two dictionary
for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    inputs, labels= inputs.to(device), labels.to(device)
    j = i +1
    Data_Dict[j] = inputs
    Label_Dict[j] = labels
    if j == netB_train_num:
        break

# Train the network 
# it need to distinguish substitute epoch and iteration epoch now
if train_flag :
    for sub_num in range(sub_epoch):
        model_B.apply(weight_reset)
        Aug_Dict = {}
        y_dict = {}
        lou = sub_num
       
        for iter_num in range(iter_epoch):
            running_loss = 0.0    
            for k in Data_Dict.keys():
                y_A = model_A(Data_Dict[k].data.detach())
                y_A = y_A.max(1)[1] 
                data = Variable(Data_Dict[k],requires_grad = True)
                optimizer_net_B.zero_grad() 
                outputs = model_B(data)
                loss = F.nll_loss(outputs, y_A)
                loss.backward()
                optimizer_net_B.step()
                running_loss += loss.item()     
            print('subtitute_epoch : ', sub_num +1, ' iteration_epoch : ', iter_num + 1,  'LOSS : ', running_loss /float(len(Data_Dict)))
        
           
        #Data Augmentation
        print('Data_Augmentation')    
        if lou >= sigma:
            sampling_list = sampling_func(lou, sigma, k_num, len(Data_Dict))
            for k in sampling_list :
                y_A = model_A(Data_Dict[k].data.detach())
                #backward_label = torch.zeros((1,10), dtype = torch.float32, device = device)
                y_A = y_A.max(1)[1] 
                #backward_label[0,y_A[0].item()] = 1
                data = Variable(Data_Dict[k].detach(),requires_grad = True)
                output = model_B(data)
                loss = F.nll_loss(output, y_A)
                loss.backward()
                #output.backward(backward_label)
                data_grad = data.grad.data
                perturbed_data = fgsm_attack(data, lamb*(-1)**int(sub_num/tau), data_grad)
                Aug_Dict[num] =perturbed_data.clone()
                num += 1
        elif lou < sigma:
            for k in Data_Dict.keys():
                 y_A = model_A(Data_Dict[k].data.detach())
                 #backward_label = torch.zeros((1,10), dtype = torch.float32, device = device)
                 y_A = y_A.max(1)[1] 
                 #backward_label[0,y_A[0].item()] = 1
                 data = Variable(Data_Dict[k].detach(),requires_grad = True)
                 output = model_B(data)
                 #output.backward(backward_label)
                 loss = F.nll_loss(output, y_A)
                 loss.backward()
                 data_grad = data.grad.data
                 perturbed_data = fgsm_attack(data, lamb*(-1)**int(sub_num/tau), data_grad)
                 Aug_Dict[num] =perturbed_data.clone()
                 num += 1
             
        print('Data_Dict length : ', len(Data_Dict))    
        Data_Dict.update(Aug_Dict)
    torch.save(net_B.state_dict(),'../data/net_B.pth')
    print('Finished Training network_B for black box adversarial attack ')

model_B.load_state_dict(torch.load('../data/net_B.pth', map_location='cpu'))
print('load pretrained based network_B data')



# test for the result which is compared  O(x') with real labels.
def test(model_A, model_B, device, test_loader, epsilon, test_set_check_count):
    count = 0
    sumA = 0
    sumB = 0
    correct = 0
    trans = 0
    model_A.eval()
    
    for  i, data in enumerate(test_loader, 0):
        if not i < test_set_check_count:
             inputs, labels = data
             inputs, labels= inputs.to(device), labels.to(device) 
             inputs.requires_grad = True
             output = model_B(inputs)
             loss = F.nll_loss(output, labels)
             model_B.zero_grad()
             loss.backward()
             inputs_grad = inputs.grad.data
             perturbed_data = fgsm_attack(inputs, epsilon, inputs_grad)
             output = model_A(perturbed_data)
             final_pred = output.max(1, keepdim=True)[1] 
             if final_pred.item() == labels.item():
                 correct += 1 
             if i %100 == 0:
                 view = perturbed_data.to('cpu').detach()
                 plt.imshow(view[0,0,:,:],cmap = 'gray')
                 plt.imsave(PATH_Result+'/result'+str(i)+'.png', view[0,0,:,:], cmap = 'gray')
    model_B.eval()
    for i, data in enumerate(test_loader, 0):
        if not i < test_set_check_count:
            inputs, labels = data
            inputs, labels= inputs.to(device), labels.to(device)     
            count +=1 
            result_vecA = model_A(inputs)
            y_A = result_vecA.max(1)[1] 
            if y_A.data == labels.data:
                sumA += 1 
            
            result_vecB = model_B(inputs)
            y_B = result_vecB.max(1)[1] 
            if y_B.data == labels.data:
                sumB += 1
            if y_B.data == y_A.data:
                trans += 1 
    print('\n \n eps : ', epsilon)
    print('the number of test set : ', count-test_set_check_count)            
    print('accuracy model A : ', sumA/count)            
    print('accuracy  model B : ', sumB/count)
    print('accuracy model, A = B : ', trans/count)
    print('missclassification transferability B to A : ', 1 - correct/count)

test_set_check_count = 0
print('test ....')
test(model_A,model_B, device, test_loader, epsilon,test_set_check_count)




