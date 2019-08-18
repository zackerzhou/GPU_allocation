#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import time
import signal
import torch
import gc


sys.path.append('/pytorch-cifar-master/')

# mode = 'cuda'
mode = 'inference'


def profile_cuda():
    DEVNULL = open(os.devnull, 'wb')
    # path_exe = ['PolyBench_exe','CUDA_samples/NVIDIA_CUDA-9.0_Samples/bin/x86_64/linux/release']
    path_exe = ['PolyBench_exe', 'CUDA_samples/NVIDIA_CUDA-9.0_Samples/bin/x86_64/linux/release']
    # file_exe = ['2DConvolution', '3DConvolution', '3mm', 'atax', 'bicg', 'gemm', 'gesummv', 'mvt', 'syr2k', 'syrk',
    #             'fdtd2d', 'correlation', 'covariance']
    # file_exe = ['2mm','gramschmidt']   # 2mm taske more than 200 seconds, gramschmidt takes 160 seconds




    # collect the metrics of benchmarks
    for i in path_exe:
        # get all the executive file names of the current folder
        file_list = os.listdir(i)
        bench_list =[]
        for m in file_list:
            if os.path.splitext(m)[1] == '':        # '' (empty string) differs from None
                bench_list.append(os.path.splitext(m)[0])
            else:
                continue
        print('The benmarks included in current folder is as follows,')
        print(bench_list)     # list includes the names of all benchmarks

        # for j in file_exe:
        for j in bench_list:
            print('Profiling of *** %s ---> %s ***' % (i, j))
            # p1 = subprocess.Popen(
            #     'nvprof --csv --log-file ./GPU_metrics/3_category_metrics_%s_%s.csv  --print-gpu-trace --profile-child-processes  --metrics local_load_transactions,dram_utilization,l2_utilization,sm_efficiency,ipc,dram_read_throughput,dram_write_throughput,dram_read_transactions,dram_write_transactions,tex_cache_transactions,tex_cache_hit_rate,tex_cache_throughput,global_hit_rate,achieved_occupancy,l2_read_transactions,l2_write_transactions,l2_tex_read_hit_rate,l2_tex_write_hit_rate,l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput  ./%s/%s' % (
            #         j, '%p', i, j),
            #     close_fds=True,
            #     shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)

            p1 = subprocess.Popen(
                'nvprof --csv --log-file ./GPU_metrics/all_category_metrics_%s_%s.csv  --print-gpu-trace --profile-child-processes  --metrics all  ./%s/%s' % (
                    j, '%p', i, j),
                close_fds=True,
                shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)


            # p2 = subprocess.Popen('nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,nounits --loop-ms=50 --filename=./GPU_status/3_category_%s_status.csv' % (j), close_fds=True,
            #         shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)


            while 1:
                ret = subprocess.Popen.poll(p1)
                if ret == 0:
                    print('*** Normal exit ***')
                    break
                elif ret is None:
                    pass
                elif ret > 0:
                    print('*** Abnormal exit ***')
                    break

            # os.killpg(os.getpgid(p2.pid), signal.SIGTERM)
            print(' *** Profiling is completed *** ')
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)


    # collect the execution time of the kernels of the benchmarks
    # for i in path_exe:
    #     for j in file_exe:
        for j in bench_list:
            print('Profiling of *** %s ---> %s *** execution time' % (i, j))
            p1 = subprocess.Popen(
                'nvprof --csv --log-file ./GPU_duration/all_category_duration_%s_%s.csv --print-gpu-trace --profile-child-processes ./%s/%s' % (j, '%p', i, j), shell=True, preexec_fn=os.setsid)

            p2 = subprocess.Popen(
                'nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,nounits --loop-ms=50 --filename=./GPU_status/all_category_%s_status.csv' % (
                j), close_fds=True,
                    shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)

            while 1:
                ret = subprocess.Popen.poll(p1)
                if ret == 0:
                    print('*** Normal exit ***')
                    break
                elif ret is None:
                    pass
                elif ret > 0:
                    print('*** Abnormal exit ***')
                    break

            os.killpg(os.getpgid(p2.pid), signal.SIGTERM)
            print(' *** Profiling is completed *** ')
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)


def profile_inference():
    DEVNULL = open(os.devnull, 'wb')
    # [epoch, batch, job, net]
    # Net: 1) VGG 2) ResNet 3) GoogleNet 4) DenseNet 5) MobileNet
    # task = ('1', '64', 'test', 'VGG')

    epoch = '1'

    inference_exe = [(epoch, '1', 'test', 'VGG'), (epoch, '1', 'test', 'ResNet'), (epoch, '1', 'test', 'GoogleNet'),
                     (epoch, '1', 'test', 'DenseNet'), (epoch, '1', 'test', 'MobileNet'),
                     (epoch, '16', 'test', 'VGG'), (epoch, '16', 'test', 'ResNet'), (epoch, '16', 'test', 'GoogleNet'),
                     (epoch, '16', 'test', 'DenseNet'), (epoch, '16', 'test', 'MobileNet'),
                     (epoch, '32', 'test', 'VGG'), (epoch, '32', 'test', 'ResNet'), (epoch, '32', 'test', 'GoogleNet'),
                     (epoch, '32', 'test', 'DenseNet'), (epoch, '32', 'test', 'MobileNet'),
                     (epoch, '64', 'test', 'VGG'), (epoch, '64', 'test', 'ResNet'), (epoch, '64', 'test', 'GoogleNet'),
                     (epoch, '64', 'test', 'DenseNet'), (epoch, '64', 'test', 'MobileNet')]

    for i in inference_exe:
        print('Profiling of *** %s of %s, batch: %s ***' % (i[2], i[3], i[1]))

        # p1 = subprocess.Popen(
        #     'nvprof --csv --log-file ./GPU_metrics/all_category_metrics_%s_%s_%s_%s.csv  --print-gpu-trace --profile-child-processes  --metrics all  python ./pytorch-cifar-master/main_arg.py  --epoch %s --batch %s --job %s --net %s' % (
        #         i[2], i[3], i[1], '%p', i[0], i[1], i[2], i[3]), close_fds=True,
        #     shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)

        # Version 2
        p1 = subprocess.Popen(
            'nvprof --csv --log-file ./GPU_metrics/all_category_metrics_%s_%s_%s_%s_%s.csv  --print-gpu-trace --profile-child-processes  --metrics all  python ./pytorch-cifar-master/main_arg.py  --epoch %s --batch %s --job %s --net %s' % (
                i[2], i[3], i[1], i[0], '%p', i[0], i[1], i[2], i[3]), close_fds=True,
            shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)

        # str_temp = '-'
        # name = str_temp.join(i)
        # p2 = subprocess.Popen(
        #     'nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,nounits --loop-ms=50 --filename=./GPU_status/%s_status.csv' % (
        #     name), close_fds=True,
        #     shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)

        while 1:
            ret = subprocess.Popen.poll(p1)
            if ret == 0:
                break
            elif ret is None:
                pass

        # os.killpg(os.getpgid(p2.pid), signal.SIGTERM)
        print(' *** Profiling is completed *** ')
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)

    # collect the execution time of the kernels of the benchmarks
    for i in inference_exe:
        str_temp = '-'
        name = str_temp.join(i)
        print('Profiling of *** %s *** execution time' % (name))

        # p1 = subprocess.Popen(
        #     'nvprof --csv --log-file ./GPU_duration/all_category_duration_%s_%s.csv --print-gpu-trace --profile-child-processes python ./pytorch-cifar-master/main_arg.py  --epoch %s --batch %s --job %s --net %s' % (
        #     name, '%p', i[0], i[1], i[2], i[3]), close_fds=True,
        #     shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)

        # Version 2
        p1 = subprocess.Popen(
            'nvprof --csv --log-file ./GPU_duration/all_category_duration_%s_%s_%s_%s_%s.csv --print-gpu-trace --profile-child-processes python ./pytorch-cifar-master/main_arg.py  --epoch %s --batch %s --job %s --net %s' % (
                i[2], i[3], i[1], i[0], '%p', i[0], i[1], i[2], i[3]), close_fds=True,
            shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)

        # p2 = subprocess.Popen(
        #     'nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,nounits --loop-ms=50 --filename=./GPU_status/all_category_%s_status.csv' % (
        #     name), close_fds=True,
        #     shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)

        # Version 2
        p2 = subprocess.Popen(
            'nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,nounits --loop-ms=50 --filename=./GPU_status/all_category_%s_%s_%s_%s_status.csv' % (
                i[2], i[3], i[1], i[0]), close_fds=True,
            shell=True, preexec_fn=os.setsid, stdout=DEVNULL, stderr=subprocess.STDOUT)

        while 1:
            ret = subprocess.Popen.poll(p1)
            if ret == 0:
                break
            elif ret is None:
                pass

        os.killpg(os.getpgid(p2.pid), signal.SIGTERM)
        print(' *** Profiling is completed *** ')
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)



if __name__ == '__main__':
    if mode == 'cuda':
        profile_cuda()
    elif mode == 'inference':
        profile_inference()
