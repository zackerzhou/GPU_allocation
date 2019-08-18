#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import time
import csv
from itertools import combinations_with_replacement
import torch

sys.path.append('/pytorch-cifar-master/')

if __name__ == '__main__':
    DEVNULL = open(os.devnull, 'wb')
    bench = 'PolyBench_exe'
    n= '3mm'
    cnt = 0
    s = 0
    o_1 = 0
    l = 11
    total_exec = 0
    while (cnt < l):
        print('loop of task : %d' % (cnt))

        if cnt < l:
            s = 1
            start_time = time.time()
            p1 = subprocess.Popen(
                './%s/%s.exe' % (bench, n), close_fds=True,
                shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)


        while 1:
            ret1 = subprocess.Popen.poll(p1)

            if ret1 == 0:
                cnt += 1
                end_time = time.time()
                exec_time = end_time - start_time
                total_exec += exec_time
                print('Execution time of task %s is %f' % (n, exec_time))  # ********
                # o_1 = int(cnt_1 >= l_1)
                break
            elif ret1 is None:
                pass