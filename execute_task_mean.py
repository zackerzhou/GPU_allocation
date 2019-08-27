#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import time
import csv
from itertools import combinations_with_replacement
import torch
import gc
import re
from visdom import Visdom


sys.path.append('/pytorch-cifar-master/')

# mode = 'one'
mode = 'two'

# one_exec = 'cuda'

# co_exec = 'cuda'
co_exec = 'nn'

# path_exe = ['PolyBench_exe', 'pytorch-cifar-master']
path_exe = ['PolyBench_exe', 'pytorch-cifar-master', 'CUDA_samples/NVIDIA_CUDA-9.0_Samples/bin/x86_64/linux/release']

poly_exe = ['2DConvolution', '3DConvolution', '3mm', 'atax', 'bicg', 'gemm', 'gesummv', 'mvt', 'syr2k', 'syrk',
            'fdtd2d', 'correlation', 'covariance']

# poly_exe = ['gesummv', 'mvt', 'syr2k', 'syrk', 'fdtd2d', 'correlation', 'covariance']
# poly_exe = []

example_exe = ['cdpBezierTessellation', 'simpleCallback', 'template', 'newdelete', 'warpAggregatedAtomicsCG',
               'cppIntegration', 'MersenneTwisterGP11213', 'reductionMultiBlockCG', 'matrixMulDynlinkJIT', 'cudaOpenMP',
               'simpleMultiCopy', 'fp16ScalarProduct', 'quasirandomGenerator', 'systemWideAtomics', 'boxFilterNPP',
               'alignedTypes', 'StreamPriorities', 'shfl_scan', 'p2pBandwidthLatencyTest', 'simpleCooperativeGroups',
               'conjugateGradient', 'simpleCubemapTexture', 'vectorAdd', 'FilterBorderControlNPP', 'asyncAPI',
               'reduction', 'SobolQRNG', 'lineOfSight', 'MC_SingleAsianOptionP', 'ptxjit', 'cdpSimpleQuicksort',
               'simpleVoteIntrinsics', 'simpleTemplates', 'scan', 'simpleZeroCopy', 'radixSortThrust', 'inlinePTX',
               'cdpAdvancedQuicksort', 'cppOverload', 'interval', 'simpleStreams', 'binomialOptions', 'simplePrintf',
               'simpleOccupancy', 'cdpSimplePrint', 'conjugateGradientMultiBlockCG', 'MC_EstimatePiInlineP',
               'transpose', 'cdpQuadtree', 'simpleCUFFT_callback', 'simplePitchLinearTexture', 'convolutionTexture',
               'matrixMul', 'cannyEdgeDetectorNPP', 'simpleIPC', 'clock', 'BlackScholes', 'FDTD3d',
               'simpleLayeredTexture', 'sortingNetworks', 'concurrentKernels', 'threadFenceReduction', 'simpleMultiGPU',
               'convolutionSeparable', 'mergeSort', 'histogram', 'simpleSeparateCompilation', 'fastWalshTransform',
               'simpleAtomicIntrinsics', 'scalarProd']

# example_exe = ['scalarProd']

# poly_exe = []
# poly_exe = ['2mm','gramschmidt']
epoch = '32'

inference_exe = [(epoch, '1', 'test', 'VGG'), (epoch, '1', 'test', 'ResNet'), (epoch, '1', 'test', 'GoogleNet'),
                 (epoch, '1', 'test', 'DenseNet'), (epoch, '1', 'test', 'MobileNet'),
                 (epoch, '16', 'test', 'VGG'), (epoch, '16', 'test', 'ResNet'), (epoch, '16', 'test', 'GoogleNet'),
                 (epoch, '16', 'test', 'DenseNet'), (epoch, '16', 'test', 'MobileNet'),
                 (epoch, '32', 'test', 'VGG'), (epoch, '32', 'test', 'ResNet'), (epoch, '32', 'test', 'GoogleNet'),
                 (epoch, '32', 'test', 'DenseNet'), (epoch, '32', 'test', 'MobileNet'),
                 (epoch, '64', 'test', 'VGG'), (epoch, '64', 'test', 'ResNet'), (epoch, '64', 'test', 'GoogleNet'),
                 (epoch, '64', 'test', 'DenseNet'), (epoch, '64', 'test', 'MobileNet')]

# inference_exe = []

# Version 1
# one_task_time_csv = 'one_task_time_mean.csv'
# two_task_time_csv = 'two_task_time_mean.csv'
# align_two_task_time_csv = 'align_two_task_time_mean.csv'

# Version 2
one_task_time_csv = 'all_category_one_task_time_mean.csv'
two_task_time_csv = 'all_category_two_task_time_mean.csv'
align_two_task_time_csv = 'all_category_align_two_task_time_mean.csv'
align_two_task_time_gap_csv = 'all_category_align_two_task_time_gap_mean.csv'

LOOP = 10
LOOP_TWO = 4

# Do not remove this function
# find the valid benchmarks (temporary function)
# def list_cuda_example():
#     file_list = os.listdir('CUDA_samples/NVIDIA_CUDA-9.0_Samples/bin/x86_64/linux/release')
#     bench_list = []
#     for m in file_list:
#         if os.path.splitext(m)[1] == '':
#             bench_list.append(os.path.splitext(m)[0])
#         else:
#             continue
#     print('The benmarks included in current folder is as follows,')
#     print(bench_list)  # list includes the names of all benchmarks
#
#     file_name = 'total_weighted_metrics_all_category.csv'
#     f = open('./%s' % (file_name), 'r')
#     csv_reader = csv.reader(f)
#     valid_bench_list = []
#     for row in csv_reader:
#         valid_bench_list.append(row[0])
#
#     ret = [i for i in valid_bench_list if i in bench_list]
#     f.close()
#
#     file_name = 'list_valid_benchmark_in_cuda_examples.csv'
#     f = open(file_name, 'w')
#     csv_w = csv.writer(f)
#     csv_w.writerow([ret])
#     f.close()
#
#     return ret


def execute_one_mean():
    DEVNULL = open(os.devnull, 'wb')

    # ***** execute CUDA *****
    # bench = 'PolyBench_exe'
    # for i in poly_exe:
    #     loop_time, t = 0, 0
    #     ave_time = 0
    #     while (loop_time < LOOP):
    #         # ****** time: start *******
    #         start_time = time.time()
    #         print('Loop: %d, Executing of *** %s *** in %s' % (loop_time, i, bench))
    #         p1 = subprocess.Popen(
    #             './%s/%s' % (bench, i), close_fds=True,
    #             shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)
    #
    #         while 1:
    #             ret = subprocess.Popen.poll(p1)
    #             if ret == 0:
    #                 break
    #             elif ret is None:
    #                 pass
    #
    #         end_time = time.time()
    #         exec_time = (end_time - start_time)
    #         print('execution time of the task is %f' % exec_time)
    #
    #         loop_time += 1  # number of loops
    #         t += exec_time  # total execution of the benchmark (multiple loops)
    #
    #         ave_time = float(t / LOOP)
    #
    #     print('Job: %s, Exec: %d, total time: %f, average time: %f' % (i, loop_time, t, ave_time))
    #     f = open(one_task_time_csv, 'a+')
    #     csv_w = csv.writer(f)
    #     csv_w.writerow([i, ave_time, t, loop_time])
    #     time.sleep(2)
    #     f.close()

    # ***** execute CUDA examples *****
    # bench = 'CUDA_samples/NVIDIA_CUDA-9.0_Samples/bin/x86_64/linux/release'
    # for i in example_exe:
    #     loop_time, t = 0, 0
    #     ave_time = 0
    #     while (loop_time < LOOP):
    #         # ****** time: start *******
    #         start_time = time.time()
    #         print('Loop: %d, Executing of *** %s *** in %s' % (loop_time, i, bench))
    #         p1 = subprocess.Popen(
    #             './%s/%s' % (bench, i), close_fds=True,
    #             shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)
    #
    #         while 1:
    #             ret = subprocess.Popen.poll(p1)
    #             if ret == 0:
    #                 break
    #             elif ret is None:
    #                 pass
    #
    #         end_time = time.time()
    #         exec_time = (end_time - start_time)
    #         print('execution time of the task is %f' % exec_time)
    #
    #         loop_time += 1  # number of loops
    #         t += exec_time  # total execution of the benchmark (multiple loops)
    #
    #         ave_time = float(t / LOOP)
    #
    #     print('Job: %s, Exec: %d, total time: %f, average time: %f' % (i, loop_time, t, ave_time))
    #     f = open(one_task_time_csv, 'a+')
    #     csv_w = csv.writer(f)
    #     csv_w.writerow([i, ave_time, t, loop_time])
    #     time.sleep(2)
    #     f.close()

    # ***** execute inference program *****
    bench = 'pytorch-cifar-master'
    for i in inference_exe:
        loop_time, t = 0, 0
        ave_time = 0
        str_i = '-'
        while (loop_time < LOOP):
            # ****** time: start *******
            start_time = time.time()
            print('Loop:%d, Executing of *** %s ***, epoch:%s,batch:%s,mode:%s' % (loop_time, i[3], i[0], i[1], i[2]))
            p1 = subprocess.Popen(
                'python ./%s/main_arg.py  --epoch %s --batch %s --job %s --net %s' % (bench, i[0], i[1], i[2], i[3]),
                close_fds=True,
                shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)

            while 1:
                ret = subprocess.Popen.poll(p1)
                if ret == 0:
                    torch.cuda.empty_cache()
                    break
                elif ret is None:
                    pass

            end_time = time.time()
            exec_time = (end_time - start_time)

            loop_time += 1
            t += exec_time

            ave_time = float(t / LOOP)

        print('Job: %s, Exec: %d, total time: %f, average time: %f' % ("i", loop_time, t, ave_time))
        f = open(one_task_time_csv, 'a+')
        csv_w = csv.writer(f)

        # Version 1
        # csv_w.writerow([str_i.join(i), ave_time, t, loop_time])

        # Version 2
        csv_w.writerow([i[2]+'_'+i[3]+'_'+i[1], ave_time, t, loop_time])

        time.sleep(2)
        f.close()


def execute_two_mean():
    DEVNULL = open(os.devnull, 'wb')
    # read the loop time and execute time from "one_task_time.csv"
    # format ---> [name of benchmark, execution time, loop time]
    dict_one = {}
    f = open(one_task_time_csv, 'r')
    csv_r = csv.reader(f)
    for p in csv_r:
        name = p[0]
        dict_one[name] = p  # {'name':['name', exec_time, loop_time]}
    f.close()

    # emerge the inference_exe's name
    # ('32','64','test','MobileNet') ---> '32-64-test-MobileNet'
    str_i = '-'
    inference_exe_join = []
    for i in inference_exe:
        # Version 1
        # inference_exe_join.append(str_i.join(i))

        # Version
        inference_exe_join.append(i[2]+'_'+i[3]+'_'+i[1])

    # poly_exe + inference_exe_join
    # ['2DConvolution', '3DConvolution', ...] + ['32-1-test-VGG', '32-16-test-VGG', ...]
    # em_exe = poly_exe + inference_exe_join

    # Version 1
    # em_exe = poly_exe + inference_exe

    # Version 2
    em_exe = poly_exe + inference_exe + example_exe
    # execute two tasks in order
    for p in combinations_with_replacement(em_exe, 2):
        torch.cuda.empty_cache()
        i, j = p[0], p[1]
        print('Execute *** %s *** & *** %s ***' % (i, j))
        c_1, c_2 = 0, 0  # 0: PolyBench_exe, 1: inference_exe, 2: example_exe
        # verify the category of the tasks
        if i in poly_exe:
            print('task 1 in poly_exe')
            c_1 = 0
            n_1, a_1, t_1, l_1 = dict_one[i]  # obtain the name, exectuion time, loop time of task 1
        elif i in inference_exe:
            print('task 1 in inference_exe')
            c_1 = 1
            str_temp = '-'
            # n_1, a_1, t_1, l_1 = dict_one[str_temp.join(i)]
            n_1, a_1, t_1, l_1 = dict_one[i[2]+'_'+i[3]+'_'+i[1]]
        elif i in example_exe:
            print('task 1 in example_exe')
            c_1 = 2
            n_1, a_1, t_1, l_1 = dict_one[i]
        else:
            n_1, a_1, t_1, l_1 = 0, 0, 0, 0
            print('task 1 do not belong to any category')
            exit(0)

        if j in poly_exe:
            print('task 2 in poly_exe')
            c_2 = 0
            n_2, a_2, t_2, l_2 = dict_one[j]  # obtain the name, exectuion time, loop time of task 1
        elif j in inference_exe:
            print('task 2 in inference_exe')
            c_2 = 1
            str_temp = '-'
            # n_2, a_2, t_2, l_2 = dict_one[str_temp.join(j)]
            n_2, a_2, t_2, l_2 = dict_one[j[2]+'_'+j[3]+'_'+j[1]]
        elif j in example_exe:
            print('task 2 in example_exe')
            c_2 = 2
            n_2, a_2, t_2, l_2 = dict_one[j]
        else:
            n_2, a_2, t_2, l_2 = 0, 0, 0, 0
            print('task 2 do not belong to any category')
            exit(0)

        loop_time = 0
        total_exec_1, total_exec_2 = 0, 0
        exec_time_1, exec_time_2 = 0, 0
        ave_time_1, ave_time_2 = 0, 0

        while (loop_time < LOOP_TWO):
            print('loop: %d' % loop_time)
            s_1, s_2 = 0, 0
            # execute task 1
            start_time_1 = time.time()
            if c_1 == 0:
                print('task 1 is in poly_exe')
                bench1 = 'PolyBench_exe'
                p1 = subprocess.Popen(
                    './%s/%s' % (bench1, n_1), close_fds=True,
                    shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)
            elif c_1 == 1:
                print('task 1 is in inference_exe')
                bench1 = 'pytorch-cifar-master'
                p1 = subprocess.Popen(
                    'python ./%s/main_arg.py --epoch %s --batch %s --job %s --net %s' % (
                    bench1, i[0], i[1], i[2], i[3]), close_fds=True,
                    shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)
            elif c_1 == 2:
                print('task 1 is in example_exe')
                bench1 = 'CUDA_samples/NVIDIA_CUDA-9.0_Samples/bin/x86_64/linux/release'
                p1 = subprocess.Popen(
                    './%s/%s' % (bench1, n_1), close_fds=True,
                    shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)
            else:
                print('c_1 error, exit')
                exit(0)

            # execute task 2
            start_time_2 = time.time()
            if c_2 == 0:
                print('task 2 is in poly_exe')
                bench2 = 'PolyBench_exe'
                p2 = subprocess.Popen(
                    './%s/%s' % (bench2, n_2), close_fds=True,
                    shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)
            elif c_2 == 1:
                print('task 2 is in inference_exe')
                bench2 = 'pytorch-cifar-master'
                p2 = subprocess.Popen(
                    'python ./%s/main_arg.py --epoch %s --batch %s --job %s --net %s' % (
                        bench2, j[0], j[1], j[2], j[3]), close_fds=True,
                    shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)
            elif c_2 == 2:
                print('task 2 is in example_exe')
                bench2 = 'CUDA_samples/NVIDIA_CUDA-9.0_Samples/bin/x86_64/linux/release'
                p2 = subprocess.Popen(
                    './%s/%s' % (bench2, n_2), close_fds=True,
                    shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)
            else:
                print('c_2 error, exit')
                exit(0)

            while 1:
                ret1 = subprocess.Popen.poll(p1)
                ret2 = subprocess.Popen.poll(p2)

                if ret1 == 0 and s_1 == 0:
                    s_1 = 1
                    end_time_1 = time.time()
                    exec_time_1 = (end_time_1 - start_time_1)

                if ret2 == 0 and s_2 == 0:
                    s_2 = 1
                    end_time_2 = time.time()
                    exec_time_2 = (end_time_2 - start_time_2)

                if s_1 == 1 and s_2 == 1:
                    loop_time += 1
                    total_exec_1 += exec_time_1
                    total_exec_2 += exec_time_2
                    torch.cuda.empty_cache()
                    break
                else:
                    pass

        ave_time_1 = float(total_exec_1 / LOOP_TWO)
        ave_time_2 = float(total_exec_2 / LOOP_TWO)

        f = open(two_task_time_csv, 'a+')
        csv_w = csv.writer(f)
        csv_w.writerow([n_1, n_2, a_1, ave_time_1, a_2,
                        ave_time_2])  # [name of task1, name of task2, execution time of task1(one), execution time of task1(two), execution time of task2(one), execution time of task2(two)]
        f.close()
        time.sleep(2)
        torch.cuda.empty_cache()
        gc.collect()

#
def align_execute_mean_two():
    temp = []
    f = open(two_task_time_csv, 'r')
    csv_r = csv.reader(f)
    for p in csv_r:
        temp.append(p)
    f.close()
    length = len(temp)

    print('length of rough pearson correlation list: %d' % length)
    # Build the dictionary for storing the complemented correlation
    str_i = '-'
    inference_exe_join = []
    for i in inference_exe:
        # inference_exe_join.append(str_i.join(i))
        inference_exe_join.append(i[2]+'_'+i[3]+'_'+i[1])

    dict_two = {}
    # em_exe = poly_exe + inference_exe_join
    em_exe = poly_exe + inference_exe_join + example_exe

    for p in em_exe:
        dict_two[p] = []

    # search each kind of benchmark and complement all the combinations with it
    for i in range(length):
        obj = temp[i]
        print('*** %s *** & *** %s ***' % (obj[0], obj[1]))
        group = temp[i]
        if group[0] != group[1]:
            x1, x2 = dict_two[group[0]], dict_two[group[1]]
            x1.append(group)
            x2.append([group[1], group[0], group[4], group[5], group[2], group[3]])
            dict_two[group[0]], dict_two[group[1]] = x1, x2
        else:
            x1 = dict_two[group[0]]
            x1.append(group)
            dict_two[group[0]] = x1

    # write the dictionary to csv file
    # f = open(align_two_task_time_csv, 'a+')
    # csv_w = csv.writer(f)
    #
    # for p in dict_two.keys():
    #     list_cor = dict_two[p]
    #     for i in list_cor:
    #         # print i
    #         csv_w.writerow(i)
    # f.close()

    # write the dictionary to csv file
    f = open(align_two_task_time_gap_csv, 'a+')
    csv_w = csv.writer(f)

    for p in dict_two.keys():
        list_cor = dict_two[p]
        for i in list_cor:
            csv_w.writerow([i[0],i[1],float(i[3])-float(i[2]),float(i[5])-float(i[4])])
    f.close()

if __name__ == '__main__':
    # example_exe = list_cuda_example()

    # if mode == 'one':
    #     execute_one_mean()
    # elif mode == 'two':
    #     execute_two_mean()

    # execute_one_mean()
    # execute_two_mean()
    # align_execute_mean_two()

    viz = Visdom()
    assert viz.check_connection()

    try:
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        plt.plot([x**2 for x in range(10)])
        plt.ylabel('numerical numbers')
        viz.matplot(plt)
    except BaseException as err:
        print('Skipped matplotlib example')
        print('Error message: ', err)

