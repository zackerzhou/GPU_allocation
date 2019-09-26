#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import csv
from itertools import combinations_with_replacement
import torch
import gc
from visdom import Visdom
import time


# ----------------------------------------------------------------------------------------------
# Setting & Configuration --- Start
# ----------------------------------------------------------------------------------------------
path_exe = ['PolyBench_exe', 'pytorch-cifar-master', 'CUDA_samples/NVIDIA_CUDA-9.0_Samples/bin/x86_64/linux/release']

poly_exe = ['2DConvolution', '3DConvolution', '3mm', 'atax', 'bicg', 'gemm', 'gesummv', 'mvt', 'syr2k', 'syrk',
            'fdtd2d', 'correlation', 'covariance']

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

epoch = '32'

inference_exe = [(epoch, '1', 'test', 'VGG'), (epoch, '1', 'test', 'ResNet'), (epoch, '1', 'test', 'GoogleNet'),
                 (epoch, '1', 'test', 'DenseNet'), (epoch, '1', 'test', 'MobileNet'),
                 (epoch, '16', 'test', 'VGG'), (epoch, '16', 'test', 'ResNet'), (epoch, '16', 'test', 'GoogleNet'),
                 (epoch, '16', 'test', 'DenseNet'), (epoch, '16', 'test', 'MobileNet'),
                 (epoch, '32', 'test', 'VGG'), (epoch, '32', 'test', 'ResNet'), (epoch, '32', 'test', 'GoogleNet'),
                 (epoch, '32', 'test', 'DenseNet'), (epoch, '32', 'test', 'MobileNet'),
                 (epoch, '64', 'test', 'VGG'), (epoch, '64', 'test', 'ResNet'), (epoch, '64', 'test', 'GoogleNet'),
                 (epoch, '64', 'test', 'DenseNet'), (epoch, '64', 'test', 'MobileNet')]


align_two_task_time_gap_csv = 'all_category_align_two_task_time_gap_mean.csv'

# ----------------------------------------------------------------------------------------------
# Setting & Configuration --- End
# ----------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------
# Environment --- Start
# Functionality: Find the gap between the interference socre and performance degradation
# Input: Interference score
# Output: Gap (used to construct the "Reward" function)
# ----------------------------------------------------------------------------------------------
class env_metric_sel():
    def __init__(self, inter_score, path_perf):
        self.inter_score = inter_score
        self.path_perf = path_perf

    def read_perf(self):
        pass








# ----------------------------------------------------------------------------------------------
# Environment --- End
# ----------------------------------------------------------------------------------------------