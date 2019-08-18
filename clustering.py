#!/usr/bin/python
# -*- coding: utf-8 -*-


import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy import stats
# from sklearn import preprocessing
# import math

input_file_name = 'total_metrics_test.csv'


# **************************************
# Read the data from csv file
# Each row: metrics for this kernel
# **************************************
def read_metrics():
    with open(input_file_name) as inf:
        has_reader = csv.Sniffer().has_header(inf.read())
        inf.seek(0)
        csv_reader = csv.reader(inf)
        if has_reader:
            next(csv_reader)
        array = []
        for row in csv_reader:
            array.append([float(m) for m in row])
        matrix = np.asarray(array)
    inf.close()
    return matrix

# **************************************
# Z-score to normalize the data
# axis = 0 ---> normalization for row, 1 ---> normalization for column
# **************************************
def zscore(Y):
    # Y should be np.array format
    norm_Y = stats.zscore(Y, axis=0,ddof=0)
    # norm_Y = preprocessing.scale(Y)

    f = open('zscore_total_metrics.csv', 'w')
    csv_w = csv.writer(f)
    for i in norm_Y:
        csv_w.writerow([m for m in i])
    f.close()
    return norm_Y

# **************************************
# Initialize the parameters of the models
# Shape: size of samples (# of samples, # of features/metrics)
# K: number of models
# **************************************
def init_params(shape, K):
    N ,D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    return mu, cov, alpha

# **************************************
# k th model's Gaussian distribution function
# each row i: occurrence rate of the i th sample in each model
# K: number of models
# **************************************
def phi(Y, mu_k, cov_k):

    # norm = multivariate_normal(mean=mu_k, cov=cov_k, allow_singular=True)
    norm = multivariate_normal(mean=mu_k, cov=cov_k)

    return norm.pdf(Y)

# **************************************
# Expectation step: compute the response for each sample
# Y: sample matrix, one row for each sample
# mu: each row represents mean of each feature
# cov: covariance array, alpha: array of model response
# **************************************
def getExpectation(Y, mu, cov, alpha):
    # number of samples
    N = Y.shape[0]
    # number of models
    K = alpha.shape[0]

    # To avoid to use 1 Gaussian model or 1 sample that resulsts in error, number of models or samples must be more than 1
    assert N > 1, "There must be more than one sample"
    assert K > 1, "There must be more than one Gaussian model"

    # response matrix, row: samples, column: response degree
    gamma = np.mat(np.zeros((N, K)))

    # print(cov)
    # Compute the possibility of all samples' occurrence rate, row: samples, column: model
    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])   # issues in here

    prob = np.mat(prob)

    # Compute each model's response degree for each sample
    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])

    return gamma

# **************************************
# Maximize step: iterate model parameters
# Y: sample matrix, gamma: response degree matrix
# **************************************
def maximize(Y, gamma):
    # number of samples and features
    N, D = Y.shape
    # number of models
    K = gamma.shape[1]

    # Initialize parameters
    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)

    # Update the parameters for each model
    for k in range(K):
        # Sum of the response degree of the k th model for all samples
        Nk = np.sum(gamma[:, k])
        # Update mu
        # Computation for each feature
        mu[k, :] = np.sum(np.multiply(Y, gamma[:, k]), axis=0) / Nk
        # Update cov
        cov_k = (Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k]) / Nk
        cov.append(cov_k)
        # Update alpha
        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha

# **************************************
# Data pre-processing (Normalization)
# Scale each data to the range [0, 1]
# **************************************
def scale_data(Y):
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    # print Y
    return Y


# **************************************
# GMM EM algorithms
# Given sample matrix: Y，compute parameters of models
# K: number of models
# times; number of iterations
# **************************************
def GMM_EM(Y, K, times):
    Y = scale_data(Y)
    mu, cov, alpha = init_params(Y.shape, K)


    # print(mu.shape)
    # print(cov.shape)
    # print(alpha.shape)

    for i in range(times):
        print('i = %d' % i)
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)

        # print(alpha)
        # print(alpha.shape)

    return mu, cov, alpha

if __name__ == '__main__':
    Y = read_metrics()     # read the data (metrics)

    norm_Y = zscore(Y)
    # print(norm_Y.shape)

    Y = norm_Y.T

    # print(Y.shape)
    # exit(0)
    # Y = np.loadtxt('gmm.data')
    # print(type(Y))
    # print(Y.shape)
    matY = np.matrix(Y, copy=True)
    # print(type(matY))
    # print(matY.shape)

    # number of Gaussian models
    K = 2

    # compute the parameters of GMM models
    mu, cov, alpha = GMM_EM(matY, K, 80)       # issues here

    # According to GMM model, clustering for sample data, one model for one category
    N = Y.shape[0]
    # In current model parameters, compute the matrix of response degree for all samples
    gamma = getExpectation(matY, mu, cov, alpha)
    # For each sample, compute the index of the model which has the highest response degree to be the category
    category = gamma.argmax(axis=1).flatten().tolist()[0]

    exit(0)
    # Add each sample into proper list
    class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
    class2 = np.array([Y[i] for i in range(N) if category[i] == 1])
    # class3 = np.array([Y[i] for i in range(N) if category[i] == 2])
    # class4 = np.array([Y[i] for i in range(N) if category[i] == 3])
    # Visualize the results
    plt.plot(class1[:, 0], class1[:, 1], 'rs', label="class1")
    plt.plot(class2[:, 0], class2[:, 1], 'bo', label="class2")
    # plt.plot(class3[:, 0], class3[:, 1], 'y4', label="class3")
    # plt.plot(class4[:, 0], class4[:, 1], 'gh', label="class4")
    # plt.legend(loc="best")
    # plt.title("GMM Clustering By EM Algorithm")
    plt.show()

    # Y = np.loadtxt("gmm.data")
    # matY = np.matrix(Y, copy=True)
    #
    # # 模型个数，即聚类的类别个数
    # K = 2
    #
    # # 计算 GMM 模型参数
    # mu, cov, alpha = GMM_EM(matY, K, 100)
    #
    # # 根据 GMM 模型，对样本数据进行聚类，一个模型对应一个类别
    # N = Y.shape[0]
    # # 求当前模型参数下，各模型对样本的响应度矩阵
    # gamma = getExpectation(matY, mu, cov, alpha)
    # # 对每个样本，求响应度最大的模型下标，作为其类别标识
    # category = gamma.argmax(axis=1).flatten().tolist()[0]
    # # 将每个样本放入对应类别的列表中
    # class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
    # class2 = np.array([Y[i] for i in range(N) if category[i] == 1])
    #
    # # 绘制聚类结果
    # plt.plot(class1[:, 0], class1[:, 1], 'rs', label="class1")
    # plt.plot(class2[:, 0], class2[:, 1], 'bo', label="class2")
    # plt.legend(loc="best")
    # plt.title("GMM Clustering By EM Algorithm")
    # plt.show()