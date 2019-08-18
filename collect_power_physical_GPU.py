#!/usr/bin/python
# -*- coding: utf-8 -*-
# send a message to VMs to start the training and data collection of vGPU (e.g., utilization)

import datetime
import subprocess
import os


print('host server staring...')
# s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)    # create IPv4-based TCP protocol socket
# s.bind(('155.69.146.41',101))                          # local IP, port(?)
# s.listen(5)                                             # start to listen

mode = 0                                                # 0: single task, 1: two tasks (training+testing), 2: three tasks(training+testing+encoding), 3: video processing task



# continuously start multiple tasks
# [epoch, batch, job, net]
task_1 = ('1', '1', 'test', 'MobileNet')  # training task
task_2 = ('3', '1', 'test', 'MobileNet')   # testing task
task_3 = ('3', '1', 'test', 'MobileNet')

if mode == 0:           # training
    start_total_x = datetime.datetime.now()
    p2 = subprocess.Popen('python ./pytorch-cifar-master/main_arg.py  --epoch %s --batch %s --job %s --net %s' % (task_1), shell=True, preexec_fn=os.setsid)
    # p2 = subprocess.Popen('ffmpeg -i 6.mkv -c:v h264_nvenc output.mp4', shell=True, preexec_fn=os.setsid)
elif mode == 1:         # training + testing
    start_total_x = datetime.datetime.now()
    p2 = subprocess.Popen('python ./pytorch-cifar-master/main_arg.py  --epoch %s --batch %s --job %s --net %s' % (task_1), shell=True, preexec_fn=os.setsid)
    p3 = subprocess.Popen('python ./pytorch-cifar-master/main_arg.py  --epoch %s --batch %s --job %s --net %s' % (task_2), shell=True, preexec_fn=os.setsid)
elif mode ==2:          # training + testing + video processing
    start_total_x = datetime.datetime.now()
    p2 = subprocess.Popen('python ./pytorch-cifar-master/main_arg.py  --epoch %s --batch %s --job %s --net %s' % (task_1), shell=True, preexec_fn=os.setsid)
    p3 = subprocess.Popen('python ./pytorch-cifar-master/main_arg.py  --epoch %s --batch %s --job %s --net %s' % (task_2), shell=True, preexec_fn=os.setsid)
    p4 = subprocess.Popen('ffmpeg -i 5.mkv -c:v h264_nvenc output.mp4', shell=True, preexec_fn=os.setsid)

elif mode ==3:
    start_total_x = datetime.datetime.now()
    k = 0
    while 1:
        p2 = subprocess.Popen('ffmpeg -i 4.mkv -c:v h264_nvenc output.mp4',shell=True, preexec_fn=os.setsid)
        while 1:
            ret = subprocess.Popen.poll(p2)
            if ret == 0:
                break
            elif ret is None:
                pass

        p2 = subprocess.Popen('rm -rf output.mp4',shell=True, preexec_fn=os.setsid)

        while 1:
            ret = subprocess.Popen.poll(p2)
            if ret == 0:
                break
            elif ret is None:
                pass
        k += 1
        if k >= 1:
            break
    end_total_x = datetime.datetime.now()
    exec_time_x = int((end_total_x - start_total_x).seconds)
    print('the total time of the execution: %d seconds' % exec_time_x)
elif mode == 4:
    start_total_x = datetime.datetime.now()
    p2 = subprocess.Popen('python ./pytorch-cifar-master/main_arg.py  --epoch %s --batch %s --job %s --net %s' % (task_1), shell=True, preexec_fn=os.setsid)
    p3 = subprocess.Popen('python ./pytorch-cifar-master/main_arg.py  --epoch %s --batch %s --job %s --net %s' % (task_2), shell=True, preexec_fn=os.setsid)
    p4 = subprocess.Popen('python ./pytorch-cifar-master/main_arg.py  --epoch %s --batch %s --job %s --net %s' % (task_3), shell=True, preexec_fn=os.setsid)

# **************************************************************************
# if all tasks has been started
# wait for them ending and collect the time data
# **************************************************************************

# start_total = datetime.datetime.now()
# print('*** start to collect the time of GPU execution ***')


if mode == 0:
    while 1:
        ret = subprocess.Popen.poll(p2)
        if ret == 0:
            break
        elif ret is None:
            pass
    end_total_x = datetime.datetime.now()
    exec_time_x = int((end_total_x - start_total_x).seconds)
    print('the total time of the execution: %d seconds' % exec_time_x)
elif mode == 1:
    while 1:
        ret2 = subprocess.Popen.poll(p2)
        ret3 = subprocess.Popen.poll(p3)
        if ret2 == 0 and ret3 ==0:
            break
        else:
            pass
    end_total_x = datetime.datetime.now()
    exec_time_x = int((end_total_x - start_total_x).seconds)
    print('the total time of the execution: %d seconds' % exec_time_x)
elif mode == 2:
    while 1:
        ret2 = subprocess.Popen.poll(p2)
        ret3 = subprocess.Popen.poll(p3)
        ret4 = subprocess.Popen.poll(p4)
        if ret2 == 0 and ret3 == 0 and ret4 == 0:
            break
        else:
            pass
    end_total_x = datetime.datetime.now()
    exec_time_x = int((end_total_x - start_total_x).seconds)
    print('the total time of the execution for multiple tasks: %d seconds' % exec_time_x)
    px = subprocess.Popen('rm -rf output.mp4', shell=True, preexec_fn=os.setsid)
elif mode ==4:
    while 1:
        ret2 = subprocess.Popen.poll(p2)
        ret3 = subprocess.Popen.poll(p3)
        ret4 = subprocess.Popen.poll(p4)
        if ret2 == 0 and ret3 == 0 and ret4 == 0:
            break
        else:
            pass
    end_total_x = datetime.datetime.now()
    exec_time_x = int((end_total_x - start_total_x).seconds)
    print('the total time of the execution for multiple tasks: %d seconds' % exec_time_x)


