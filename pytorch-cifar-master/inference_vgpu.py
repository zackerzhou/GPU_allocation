#!/usr/bin/python
# -*- coding: utf-8 -*-
'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import datetime
import ctypes
import random
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import time
import signal
import subprocess

lib = ctypes.cdll.LoadLibrary(None)

# ***********************
# *** Test parameters ***
# ***********************
# EPOCHES = 2
EPOCHES = 3
# EPOCHES = 40
MODEL = 0       # 1 for one network on multi-GPU, 0 for one network on one-GPU
# ***********************

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

if MODEL == 1:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('1 network on multi-GPU')
elif MODEL == 0:
    # device = 'cuda: 2' if torch.cuda.is_available() else 'cpu'
    device = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
    print('1 network on one-GPU')
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# frame_in_list = []
# frame_out_list = []
# for frame_idx, (frame_batch, frame_target) in enumerate(testloader):
#     frame_in_list.append(frame_batch)
#     frame_out_list.append(frame_target)
# len_set = len(frame_in_list)



# Model
print('==> Building model..')
# net = VGG('VGG19')        # collected_10s 4s 2s * *
# net = ResNet18()          # collected_10s 4s * *
# net = PreActResNet18()
net = GoogLeNet()         # collected_10s 4s * *
# net = DenseNet121()       # collected_10s 4s * *
# net = ResNeXt29_2x64d()
# net = MobileNet()         # collected_10s 4s * *
# net = MobileNetV2()       # collected_10s 4s
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
# @profile
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # if batch_idx >= 3:
        #     break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# @profile
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)  # loss


            test_loss += loss.item()            # tensor -> scalar
            _, predicted = outputs.max(1)       # find the predicted result

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.t7')
    #     best_acc = acc

def inference(frame_in,frame_out):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = frame_in.to(device), frame_out.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)  # loss

        torch.cuda.synchronize()
        test_loss += loss.item()  # tensor -> scalar
        _, predicted = outputs.max(1)  # find the predicted result

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return test_loss

if __name__ == '__main__':
    start_total = datetime.datetime.now()

    # lib.cuProfilerStart()
    # p = subprocess.Popen('nvidia-smi --query-gpu=fan.speed,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,nounits --id=2 --loop-ms=50 --filename=/home/zhouxin/program/GPU_modeling/data_collection/MobileNet_train/1_train_epoch_GPU_2.csv',shell=True, preexec_fn=os.setsid)

    print(test)
    loss = 0
    for epoch in range(start_epoch, start_epoch+EPOCHES):
        # train(epoch)
        test(epoch)

    # for epoch in range(start_epoch, start_epoch+EPOCHES):
    #     seed = random.randint(0,len_set)
    #
    #     if seed >= loss:
    #         seed = int(loss)
    #
    #
    #     inputs = frame_in_list[seed]
    #     outputs = frame_out_list[seed]
    #
    #     loss = inference(inputs,outputs)
    #     print('%d: %f' % (epoch, loss))

    # lib.cuProfilerStop()

    end_total = datetime.datetime.now()
    exec_time = int((end_total-start_total).seconds)
    print('the total time of the net: %d seconds' % exec_time)
    print('the average time of the net: %f seconds' % float(exec_time/EPOCHES))

    lib.cudaProfilerStop()


    # time.sleep(2)
    # os.killpg(os.getpgid(p.pid), signal.SIGTERM)

    # print('test 123')