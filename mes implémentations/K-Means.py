import csv 
import random
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

input_file = open('Country-data.csv')
reader = csv.reader(input_file)
next(reader)

k = 2
data = []
country_names = []

for country in reader:
    data.append([country[0]] + [float(feature) for feature in country[5]])
    country_names.append(country[0])

def distance(country, centroid, option):
    dist_ = sum([(value - value_)**2 for value, value_ in zip(country[1:], centroid[1:])])
    if option == 1:
        return sqrt(dist_)
    elif option == 2:
        return dist_

def dictionary(centroids):
    dict_ = {}
    for centroid in centroids:
        dict_[tuple(centroid)] = []
    return dict_

def assagning(data, dict_):
    for country in data:
        tmp = []
        tmp_ = []
        for centroid in dict_:
            dist = distance(country, centroid, 1)
            tmp.append(dist)
            tmp_.append(centroid)
        minimum = tmp.index(min(tmp))
        dict_[tuple(tmp_[minimum])].append(tuple(country))
    return dict_

def mean(list_):
    new_list = []
    [new_list.append(i[1:]) for i in list_]
    tmp = zip(*new_list)
    new_centroids = []
    for i in tmp:
        result = sum(i)/len(i)
        new_centroids.append(result)
    return new_centroids

def updating(dict_, centroids):
    centroids.clear()
    for list_, i in zip(dict_.values(), dict_):
        new_centroid = [i[0]]
        new_centroid.extend(mean(list_))
        centroids.append(new_centroid)
    return centroids

testing = []

def main_loop(data, n):
    for i in range(n):
        dict_ = dictionary(centroids)
        assign = assagning(data, dict_)
        update = updating(assign, centroids)
        if i == n-2:
            break
    dict_ = dictionary(centroids)
    assign = assagning(data, dict_)
    return assign
    
def sse(dict_, centroid):
    tmp = []
    for i in dict_:
        result = abs(distance(i, centroid, 2))
        tmp.append(result)
    return sum(tmp)

def a_b(cluster, p, option):
    dist = 0
    for j in cluster:
        if p != j:
            dist += distance(p, j, 1)
    if option == 'a':
        return dist/(len(cluster)-1)
    elif option == 'b':
        return dist/len(cluster)


centroids = []
sse_list_ = []
 
global dict_

n = 1
k = 10
while n <= k:
    centroids.clear()
    for j in range(n):
        centroids.append(data[random.randint(0, (len(data)-1))])
    dict_ = main_loop(data, 100)
    for centroid in centroids:
        sse_ = sse(dict_[tuple(centroid)], centroid)
    sse_list_.append(sse_)
    n += 1


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
        
s = shilouette(dict_)

list_ = list(range(1, k+1))
plt.scatter(list_, sse_list_)
plt.plot(list_, sse_list_)
plt.show()



"""
plt.figure(figsize = (20, 7))
for centroid in centroids:
    list_ = dict_[tuple(centroid)]
    x = [x[0] for x in list_]
    y = [y[1] for y in list_]
    plt.scatter(x, y)
    plt.scatter(centroid[0], centroid[1], s = 120, c = 'black')
    plt.xticks(rotation = 90)

plt.xlabel("Pays")
plt.ylabel("Revenu")
plt.tight_layout()
plt.show()


names = Extract(data, 0)
income = Extract(data, 1)
plt.scatter(names, income)
plt.show()"""