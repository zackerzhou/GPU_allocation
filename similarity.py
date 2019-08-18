#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import csv
import math
from itertools import combinations_with_replacement
# from scipy.spatial.distance import pdist
from scipy.stats import pearsonr


sys.path.append('/pytorch-cifar-master/')

mode = 'one'
# mode = 'two'

# one_exec = 'cuda'
one_exec = 'nn'

# co_exec = 'cuda'
co_exec = 'nn'

# path_exe = ['PolyBench_exe', 'pytorch-cifar-master']
poly_exe = ['2DConvolution', '3DConvolution', '3mm', 'atax', 'bicg', 'gemm', 'gesummv', 'mvt', 'syr2k', 'syrk',
            'fdtd2d', 'correlation', 'covariance']
# poly_exe = ['2mm','gramschmidt']

# inference_exe = ['test_VGG_1','test_ResNet_1','test_GoogleNet_1','test_DenseNet_1','test_MobileNet_1',
#                  'test_VGG_16','test_ResNet_16','test_GoogleNet_16','test_DenseNet_16','test_MobileNet_16',
#                  'test_VGG_32','test_ResNet_32','test_GoogleNet_32','test_DenseNet_32','test_MobileNet_32',
#                  'test_VGG_64','test_ResNet_64','test_GoogleNet_64','test_DenseNet_64','test_MobileNet_64']

epoch = '32'

inference_exe = [(epoch, '1', 'test', 'VGG'), (epoch, '1', 'test', 'ResNet'), (epoch, '1', 'test', 'GoogleNet'),
                 (epoch, '1', 'test', 'DenseNet'), (epoch, '1', 'test', 'MobileNet'),
                 (epoch, '16', 'test', 'VGG'), (epoch, '16', 'test', 'ResNet'), (epoch, '16', 'test', 'GoogleNet'),
                 (epoch, '16', 'test', 'DenseNet'), (epoch, '16', 'test', 'MobileNet'),
                 (epoch, '32', 'test', 'VGG'), (epoch, '32', 'test', 'ResNet'), (epoch, '32', 'test', 'GoogleNet'),
                 (epoch, '32', 'test', 'DenseNet'), (epoch, '32', 'test', 'MobileNet'),
                 (epoch, '64', 'test', 'VGG'), (epoch, '64', 'test', 'ResNet'), (epoch, '64', 'test', 'GoogleNet'),
                 (epoch, '64', 'test', 'DenseNet'), (epoch, '64', 'test', 'MobileNet')]

# input_file_name = 'summary_profile_total_weighted.csv'
output_euclidean_csv = 'euclidean_two_task.csv'
align_two_task_time_csv = 'align_two_task_time_mean.csv'


# normal Pearson correlation
# output_pearson_csv = 'pearson_two_task.csv'
# align_correlation_csv = 'align_pearson_two_task.csv'

# Pearson correlation weighted by duration of kernels
# input_file_name = 'summary_profile_total_weighted.csv'
# output_pearson_csv = 'pearson_two_task_weighted.csv'
# align_correlation_csv = 'align_pearson_two_task_weighted.csv'

# Pearson correlation weighted by duration of the kernels and average utilization of the benchmark
input_file_name = 'summary_profile_total_weighted_use_util.csv'
output_pearson_csv = 'pearson_two_task_weighted_util.csv'
align_correlation_csv = 'align_pearson_two_task_weighted_util.csv'
align_emerge_interference_performance_normalized_csv = 'align_emerge_interference_performance_normalized_util.csv'
align_emerge_interference_performance_no_normalized_csv = 'align_emerge_interference_performance_no_normalized_util.csv'


def read_profile():
    # input_file_name = 'summary_profile_total.csv'
    f = open(input_file_name,'r')
    csv_reader = csv.reader(f)
    array = []
    matrix = []
    for row in csv_reader:
        array.append([m for m in row])
    matrix = np.asarray(array)      # ******
    f.close()
    return matrix

def euclidean_dist(profile_in):
    for p in combinations_with_replacement(profile_in, 2):
        i, j = p[0], p[1]
        print("Computing the similarity of %s and %s" % (i[0], j[0]))
        c1, c2 = i[1:], j[1:]
        v1, v2 = c1.astype(np.float), c2.astype(np.float)
        # x = np.vstack([v1,v2])
        # d2 = pdist(x)
        # print(d2)
        #
        # output_csv = 'euclidean_two_task.csv'
        # f = open(output_csv, 'a+')
        # csv_w = csv.writer(f)
        # csv_w.writerow([i[0], j[0], d2])
        # f.close()


        c = (v1 -v2) ** 2
        dist = 1/(1+np.sqrt(sum(c)))
        print(dist)

        # if dist != 1:
        f = open(output_euclidean_csv, 'a+')
        csv_w = csv.writer(f)
        csv_w.writerow([i[0], j[0], dist])
        f.close()

def pearson_correlation(profile_in):
    # Compute the Pearson correlation for each two benchmarks
    for p in combinations_with_replacement(profile_in, 2):
        i, j = p[0], p[1]
        print("Computing the similarity of %s and %s" % (i[0], j[0]))
        c1, c2 = i[1:], j[1:]
        v1, v2 = c1.astype(np.float), c2.astype(np.float)
        factor, b = pearsonr(v1,v2)

        f = open(output_pearson_csv, 'a+')
        csv_w = csv.writer(f)
        csv_w.writerow([i[0], j[0], factor])
        f.close()

    # align the Pearson correlation table
    temp = []
    f = open(output_pearson_csv, 'r')
    csv_r = csv.reader(f)
    for p in csv_r:
        temp.append(p)
    f.close()
    length = len(temp)

    # print('length of rough pearson correlation list: %d' % length)
    # Build the dictionary for storing the complemented correlation
    dict_cor = {}
    em_exe = poly_exe + inference_exe

    for p in em_exe:
        dict_cor[p] = []

    # search each kind of benchmark and complement all the combinations with it
    for i in range(length):
        # obj = temp[i]
        # print('*** %s *** & *** %s ***' % (obj[0],obj[1]))
        group = temp[i]
        if group[0] != group[1]:
            x1, x2 = dict_cor[group[0]], dict_cor[group[1]]
            x1.append(group)
            x2.append([group[1], group[0], group[2]])
            dict_cor[group[0]], dict_cor[group[1]] = x1, x2
        else:
            x1 = dict_cor[group[0]]
            x1.append(group)
            dict_cor[group[0]] = x1

    # write the dictionary to csv file
    f = open(align_correlation_csv, 'a+')
    csv_w = csv.writer(f)

    for p in dict_cor.keys():
        list_cor = dict_cor[p]
        for i in list_cor:
            # print i
            csv_w.writerow(i)
    f.close()

def pearson_correlation_util(profile_in):
    for p in combinations_with_replacement(profile_in, 2):
        # Each row: name, metric, metric, metric, ..., metric, average utilization
        i, j = p[0], p[1]
        print("Computing the similarity of %s and %s" % (i[0], j[0]))
        c1, c2 = i[1:], j[1:]
        v1, v2 = c1.astype(np.float), c2.astype(np.float)
        factor, b = pearsonr(v1[1:-2],v2[1:-2])

        # f(u1, u2):
        diff = (100 - (v1[-1] + v2[-1])) / 100
        factor_util = 1 / math.exp(diff)
        print('v1:%f,v2:%f,factor_util:%f' %(v1[-1],v2[-1],factor_util))

        f = open(output_pearson_csv, 'a+')
        csv_w = csv.writer(f)
        # csv_w.writerow([i[0], j[0], factor, v1[-1], v2[-1], factor_util])
        csv_w.writerow([i[0], j[0], factor, factor_util])
        f.close()

    # align the Pearson correlation table
    temp = []
    f = open(output_pearson_csv, 'r')
    csv_r = csv.reader(f)
    for p in csv_r:
        temp.append(p)
    f.close()
    length = len(temp)

    # print('length of rough pearson correlation list: %d' % length)
    # Build the dictionary for storing the complemented correlation
    str_i = '-'
    inference_exe_join = []
    for i in inference_exe:
        inference_exe_join.append(str_i.join(i))

    dict_cor = {}
    em_exe = poly_exe + inference_exe_join


    for p in em_exe:
        dict_cor[p] = []



    # search each kind of benchmark and complement all the combinations with it
    for i in range(length):
        # obj = temp[i]
        # print('*** %s *** & *** %s ***' % (obj[0],obj[1]))
        group = temp[i]
        if group[0] != group[1]:
            x1, x2 = dict_cor[group[0]], dict_cor[group[1]]
            x1.append(group)
            # x2.append([group[1], group[0], group[2], group[4], group[3], group[5]])
            x2.append([group[1], group[0], group[2], group[3]])
            dict_cor[group[0]], dict_cor[group[1]] = x1, x2
        else:
            x1 = dict_cor[group[0]]
            x1.append(group)
            dict_cor[group[0]] = x1

    # write the dictionary to csv file
    f = open(align_correlation_csv, 'a+')
    csv_w = csv.writer(f)

    for p in dict_cor.keys():
        list_cor = dict_cor[p]
        for i in list_cor:
            # print i
            csv_w.writerow(i)
    f.close()


def emerge_performance_and_interference():
    # Prerequisite: A.csv + B.csv
    # A.csv: "align_pearson_two_task_weighted_util.csv" ---> each row: [Task1, Task2, Pearson, f(utilization ot two tasks:u1,u2)]
    # B.csv: "two_task_time_mean.csv" ---> each row: [Task1, Task2, t1_1(siloed-execution), t1_2(co-execution), t2_1(siloed-execution), t2_2(co-execution)]
    # Objective: create a csv including performance degradation & interference score

    # read A.csv
    f1 = open(align_correlation_csv, 'r')
    csv_reader = csv.reader(f1)
    matrix = []
    for row in csv_reader:
        matrix.append([m for m in row])
    cor = np.asarray(matrix)
    f1.close()
    print(len(cor))

    # read B.csv
    f2 = open(align_two_task_time_csv, 'r')
    csv_reader = csv.reader(f2)
    matrix = []
    for row in csv_reader:
        matrix.append([m for m in row])
    exec_time = np.asarray(matrix)
    f2.close()
    print(len(exec_time))


    # compute the performance degradation and the interference score
    length = len(exec_time)
    list = range(length)
    emerge = []
    for i in cor:
        n1, n2 = i[0], i[1]
        pearson, fun_util = i[2], i[3]
        inter_score = float(pearson) * 0.2 + float(fun_util) * 0.8
        # search the same pair of benchmark in B.csv to get the performance degradation
        for j in list:
            m1, m2 = exec_time[j][0],exec_time[j][1]
            # If found the same pair of benchmark in B.csv
            if n1 == m1 and n2 == m2:
                degradation = float(exec_time[j][3]) - float(exec_time[j][2])       # performance degradation
                # degradation += 10    # minus ---> plus
                emerge.append([n1, n2, inter_score, degradation])
                list.remove(j)
    # Normalization of interference score and performance degradation
    str_i = '-'
    inference_exe_join = []
    for i in inference_exe:
        inference_exe_join.append(str_i.join(i))

    dict_cor = {}
    em = poly_exe + inference_exe_join

    for p in em:
        dict_cor[p] = [[],[],[],[]]

    # arrange the data to a dictionary ---> e.g., dict_cor['3mm'] = [[names],[interferences],[performance degradations]]
    for i in emerge:
        n = i[0]
        x = dict_cor[n]
        x[0].append(i[0])   # objective benchmark's name
        x[1].append(i[1])   # pair benchmark's name
        x[2].append(i[2])   # interference score
        x[3].append(i[3])   # performance degradation
        dict_cor[n] = x

    # print(dict_cor['2DConvolution'])

    # write the dictionary (no_normalized) to csv file
    f = open(align_emerge_interference_performance_no_normalized_csv, 'a+')
    csv_w = csv.writer(f)

    for p in dict_cor.keys():
        l_n1, l_n2, l_inter, l_perf = dict_cor[p]
        for i in range(len(l_n1)):
            csv_w.writerow([l_n1[i], l_n2[i], l_inter[i], l_perf[i]])
    f.close()
    exit(0)
    for i in em:
        list_inter = dict_cor[i][2]
        list_perf = dict_cor[i][3]

        max_inter = max(list_inter)
        min_inter = min(list_inter)

        max_perf = max(list_perf)
        min_perf = min(list_perf)
        # list
        inter = np.array(list_inter)
        perf = np.array(list_perf)
        # np.array
        inter_temp = (inter - min_inter) / (max_inter - min_inter)
        perf_temp = (perf - min_perf) / (max_perf - min_perf)
        # list
        inter_normalized = inter_temp.tolist()
        perf_normalized = perf_temp.tolist()

        dict_cor[i][2] = inter_normalized
        dict_cor[i][3] = perf_normalized


    # write the dictionary to csv file
    f = open(align_emerge_interference_performance_normalized_csv, 'a+')
    csv_w = csv.writer(f)

    for p in dict_cor.keys():
        l_n1, l_n2, l_inter, l_perf = dict_cor[p]
        for i in range(len(l_n1)):
            csv_w.writerow([l_n1[i], l_n2[i], l_inter[i], l_perf[i]])
    f.close()

if __name__ == '__main__':
    profiles = read_profile()     # read profiles from csv file
    # euclidean_dist(profiles)
    #
    # pearson_correlation(profiles) # compute the pair-wise pearson correlation
    pearson_correlation_util(profiles)  # compute the pair-wise pearson correlation with utilization information

    # emerge_performance_and_interference()


