import csv
import re
import random
from math import sqrt, inf
from plotly.figure_factory import create_dendrogram
import numpy as np
import itertools

def read(file_name, seperator=','):
    data = []
    with open(file_name) as input_file:
        ptr = 1
        for row in input_file.readlines():
            if ptr != 1:
                li = row.split(seperator)
                if re.findall('"', row):
                    if type(li[0]) == str and type(li[1]) == str:
                        str_ = li[0] + " " + li[1] 
                        data.append(sum([[str_], [float(x) for x in li[2:]]], []))
                else: 
                    data.append(sum([[li[0]], [float(x) for x in li[1:]]], []))   
            ptr += 1
    return data


dataset = read('Country-data.csv')

def dictionary():
    dict_ = {}
    for id_, country in enumerate(dataset):
        dict_[id_] = [country]
    return dict_

def distance(points_1, points_2):
    res = sqrt(sum([(p - p_)**2 for p, p_ in zip(points_1[1:], points_2[1:])]))
    return res

def single_linkage(cluster_1, cluster_2):
    res = [distance(values_1, values_2) for values_1 in cluster_1 for values_2 in cluster_2]
    return min(res)    

def complete_linkage(cluster_1, cluster_2):
    res = [distance(values_1, values_2) for values_1 in cluster_1 for values_2 in cluster_2]
    return max(res)

def average_linkage(cluster_1, cluster_2):
    dist_ = [distance(values_1, values_2) for values_1 in cluster_1 for values_2 in cluster_2]
    return sum(dist_)/len(dist_)


def closest_(dict_):
    tmp = []
    tmp_ = []
    keys_ = list(dict_.keys())
    for id_1, cluster_1 in enumerate(keys_[:-1]):
        for id_2, cluster_2 in enumerate(keys_[id_1+1:]):
            dist = average_linkage(dict_[cluster_1], dict_[cluster_2])
            tmp.append(dist)
            tmp_.append((cluster_1, cluster_2))
        minimum = min(tmp)
        close = tmp_[tmp.index(minimum)]
    return close

def merging(dict_, cluster_1, cluster_2):
    new_cluster = {0 : dict_[cluster_1] + dict_[cluster_2]}
    for id_ in dict_.keys():
        if (id_ == cluster_1) | (id_ == cluster_2):
            continue
        new_cluster[len(new_cluster.keys())] = dict_[id_]
    return new_cluster

def a_b(cluster, p, option):
    dist = 0
    for j in cluster:
        if p != j:
            dist += distance(p, j)
    if option == 'a':
        if len(cluster) > 1:
            return dist/(len(cluster)-1)
        else:
            return dist/len(cluster)
    elif option == 'b':
        return dist/len(cluster)

def shilouette(dict_):
    tmp = []
    for i in dict_:
        p = random.choice(list(dict_[i]))
        tmp.append(p)
    for p in tmp:
        tmp_ = []
        for i in dict_:
            if p not in dict_[i]:
                res = a_b(dict_[i], p, 'b')
                tmp_.append(res)
            else:
                a = a_b(dict_[i], p, 'a')
        b = min(tmp_)
        s_p = (b-a)/max(a, b)
        print(s_p)

k = 3

def main_loop():
    for n in range(1, k+1):
        dict_ = dictionary()
        while len(dict_.keys()) > n:
            close = closest_(dict_)
            dict_ = merging(dict_, *close)
    return dict_

dict_ = main_loop()
      
s = shilouette(dict_)