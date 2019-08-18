#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import csv
import numpy as np
# --------------------------------------------------------------------------------
# Check whether the metric was generated or not
# if yes, there will be no "Error" occurred before retrieving "Device", return 0
# if no, there will be a "Error" occurred, return 1
# --------------------------------------------------------------------------------
def check_metric_generation(path, file_name):
    f = open('./%s/%s' % (path, file_name), 'r')
    csv_reader = csv.reader(f)
    p = re.compile(r'(.*)(Error)(.*)')
    p_1 = re.compile(r'(.*)(No events/metrics)(.*)')
    q = re.compile(r'(Device)')
    for i in csv_reader:
        m = p.match(i[0])       # check 'Error'
        m_1 = p_1.match(i[0])
        if m == None and m_1 == None:
            # print('No error')
            m = q.match(i[0])   # check 'Device'
            if m == None:
                continue
            else:
                # print('Device detected')
                f.close()
                return 0
        else:
            # print('Error detected!!!')
            f.close()
            return 1


def benchmark_list(path):
    file_list = os.listdir(path)
    list = []
    list_full = []
    # Regular expression to find out the name of each benchmark
    # name of benchmark: all_category_metrics_xxxx_12314.csv, xxxx is the executable name of the benchmark
    pattern = re.compile(r'(all_category_metrics)\_(\S+)\_([0-9]+)')
    for i in file_list:
        name = i
        print(name)
        match = pattern.match(name)
        if match == None:
            # print('*** none matched ***')
            continue
        # print('matched')
        bench_name = match.group(2)
        # check whether the metrics are successfully generated for this benchmark, if not, skip this benchmark
        err = check_metric_generation(path, i)
        if err == 1:
            print('Not available')
        else:
            print('avalibale')
            list.append(bench_name)
            list_full.append(i)
    return list, list_full


# Retrieve the time file name of the benchmark
def time_rate_computation(time_path, name_full, k_list_metric):
    print(name_full)
    f = open('./%s/%s' % (time_path, name_full), 'r')
    csv_reader = csv.reader(f)
    tab = []

    p = re.compile(r'(Start)')
    en = 0

    t = 0.0     # total execution time of the benchmark
    d = 0.0     # duration of one kernel

    for row in csv_reader:
        if en == 0:
            m = p.match(row[0])
            if m != None:
                en = 1
            else:
                continue
        if en == 1:
            tab.append([row[-1], row[0],row[1]])
    tab_array = np.asarray(tab)

    time_array = tab_array[2:, 1:]
    k_list_time = tab_array[2:, 0]


    # complement the "None" cell in time array
    for i in range(time_array.shape[0]):
        for j in range(time_array.shape[1]):
            if time_array[i,j] == '':
                time_array[i,j] = '0.0'

    time = time_array.astype(float)

    # compute the total execution time of the benchmark
    # find time of the kernel ending latest, i.e., max(start time + duration)
    temp = time[:,0] + time[:,1]
    idx = np.argmax(temp, 0)

    t = time[idx, 0] + time[idx, 1] - time[0, 0]
    print('total time: %f' % t)

    f.close()

    # compare the lists of kernel(time) and kernel(metric)
    # compute the time occupancy of each valid kernel
    rate = []
    suc = 0

    # List of special char
    spec_char = ".,*[]()#$%^&_{}"
    start = 0
    l = len(k_list_time)

    for i in range(len(k_list_metric)):
        string = k_list_metric[i]

        # add '\' for each special char
        # e.g., 'compute*' ---> 'compute\*'
        # following is a nice solution found from blog
        spec_char_dict = dict([(c,'\\'+c) for c in spec_char])
        new_string = str.join('', [ spec_char_dict.get(c, c) for c in string ])
        # print(new_string)

        q = re.compile(r'(%s)' % new_string)

        # search the corresponding time occupancy rate from time list of kernels
        for j in range(start,l):
            n = q.match(k_list_time[j])
            if n == None:
                start += 1
                continue
            else:
                start += 1
                suc = 1
                time_rate = time[j, 1] / t
                rate.append(time_rate)
                # print('rate: %f' % time_rate)
                break
        if suc != 1:
            print('error: did not find the kernel in time list')
            exit(0)
        else:
            pass
    print('successfully retrieve the time occupancy rate of each kernel')
    rate_array = np.array(rate)[:,np.newaxis]

    return rate_array


# --------------------------------------------------------------------------------
# Benchmark may have thousands of kernels
# This function is to use the time occupancy rate to weight each kernel
# Finally, all the weighted kernels are combined to one which represents the
# behavior of this benchmark
# Format of metrics: |Device, Contex, Stream, Kernel, "inst_per_warp", ... ...|
#                        0       1       2       3           4
# --------------------------------------------------------------------------------
def metric_assemble_with_time_occupancy_rate(metrics_path, time_path, name, name_full):
    # read a benchmark's metrics (all kernels)
    tab = []
    f = open('./%s/%s' % (metrics_path, name_full), 'r')
    csv_reader = csv.reader(f)
    p = re.compile(r'(Device)')
    en = 0
    for row in csv_reader:
        if en == 0:
            m = p.match(row[0])
            if m != None:
                en = 1
                tab.append(row)
            else:
                continue
        elif en == 1:
            tab.append(row)

    tab_array = np.asarray(tab)     # list -> array

    # remove row "Item", "Unit"
    # remove column "Device", "Contex", "Stream", "Kernel"
    # store the kernel list of the benchmark
    metrics_array = tab_array[2:, 4:]
    kernel_list = tab_array[2:,3]

    x = metrics_array.shape[0]
    y = metrics_array.shape[1]

    # scan metrics_array to convert the format of each metric
    p = re.compile(r'(\S*)\s\(([0-9]+)\)')
    for i in range(x):
        for j in range(y):
            cell = metrics_array[i,j]      # Verification, "Low(1) -> 1", "Idle(0) -> 0"
            m = p.match(cell)
            if m == None:                  # if cell do not contain "Alphabet"
                continue
            else:
                metrics_array[i,j] = m.group(2)

    # convert the overflow array to a valid number
    print("benchmark: %s" % name)
    for i in range(metrics_array.shape[0]):
        for j in range(metrics_array.shape[1]):
            if metrics_array[i,j] == '<OVERFLOW>':
                metrics_array[i,j] = '0.0'
                # print('detected')


    metrics = metrics_array.astype(float)

    print("Shape of metrics array: (%d, %d)" % metrics.shape)
    f.close()

    # get the file list of time duration of all benchmarks
    time_list = os.listdir(time_path)

    # get the corresponding full name of time file
    time_name = []
    q = re.compile(r'(all_category_duration)\_(%s)\_([0-9]+)' % name)
    for i in time_list:
        m = q.match(i)
        if m == None:
            continue
        else:
            time_name = i
            break

    # read a benchmark's time information (all kernels) to calculate the time occupancy rate of each kernel
    rates = time_rate_computation(time_path, time_name, kernel_list)
    print('len(metrics): %d' % metrics.shape[0])
    print('len(rate): %d' % len(rates))
    # print(rates)
    if metrics.shape[0] != len(rates):
        return 1, ''
    else:
        weighted_metrics = metrics * rates
        mean_metrics = np.mean(weighted_metrics,axis=0)

        return 0, mean_metrics


def write_weighted_metrics(file_name, weighted_metrics):
    f = open(file_name, 'w')
    csv_w = csv.writer(f)

    for row in weighted_metrics:
        csv_w.writerow(row)
    f.close()
    pass


def compute_mean_util():
    pass
    return 0

if __name__ == '__main__':
    metrics_path = 'GPU_metrics'
    time_path = 'GPU_duration'

    # get the list of benchmarks with available metrics
    l, l_f = benchmark_list(metrics_path)
    print("Number of available benchmark: %d" % len(l_f))

    # read metrics from each available benchmark as well as change the strings to floating numbers
    w_m = []

    for i in range(len(l_f)):
        status, temp_array = metric_assemble_with_time_occupancy_rate(metrics_path, time_path, l[i], l_f[i])
        if status == 1:
            print('%s not available to get the mean metrics' % l[i])
        else:
            temp_array = temp_array.tolist()
            row = [l[i]] + temp_array
            # print(row)
            w_m.append(row)
            pass
        # exit(0)

    print('num of available benchmarks: %d' % len(w_m))
    print('num of metrics for each benchmark: %d' % len(w_m[0]))

    # write to csv file
    file_name = 'total_weighted_metrics_all_category.csv'
    write_weighted_metrics(file_name, w_m)
