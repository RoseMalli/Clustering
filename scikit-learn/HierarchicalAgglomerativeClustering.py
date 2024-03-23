import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster import hierarchy

data = pd.read_csv('Country-data.csv')

income = data[['income']]

silhouette = []

for i in range(2, 11):
    hac = AgglomerativeClustering(n_clusters = i, metric = 'euclidean', linkage = 'ward')
    silhouette.append(silhouette_score(income, hac.fit_predict(income)))

hac = AgglomerativeClustering(n_clusters = 2, metric = 'euclidean', linkage = 'ward')
hac_ = hac.fit_predict(income)
data['cluster'] = hac_

plt.figure(figsize = (20, 7))
plt.title('Agglomerative Clustering', size = 18, fontweight = 'bold')
plt.scatter(data['country'], income, c = data['cluster'], cmap = 'plasma')
plt.xticks(rotation = 90)
plt.xlabel('Country', fontweight = 'bold')
plt.ylabel('Income', fontweight = 'bold')
plt.tight_layout()
plt.show()

plt.figure(figsize = (20, 7))
plt.title('Agglomerative Clustering (Linkage = Single)', size = 18, fontweight = 'bold')
hierarchy.dendrogram(hierarchy.linkage(income, method = 'single'), labels = data['country'].tolist())
plt.xlabel('Country', fontweight = 'bold')
plt.ylabel('Income', fontweight = 'bold')
plt.tight_layout()
plt.show()

plt.figure(figsize = (20, 7))
plt.title('Agglomerative Clustering (Linkage = Complete)', size = 18, fontweight = 'bold')
hierarchy.dendrogram(hierarchy.linkage(income, method = 'complete'), labels = data['country'].tolist())
plt.xlabel('Country', fontweight = 'bold')
plt.ylabel('Income', fontweight = 'bold')
plt.tight_layout()
plt.show()

plt.figure(figsize = (20, 7))
plt.title('Agglomerative Clustering (Linkage = Average)', size = 18, fontweight = 'bold')
hierarchy.dendrogram(hierarchy.linkage(income, method = 'average'), labels = data['country'].tolist())
plt.xlabel('Country', fontweight = 'bold')
plt.ylabel('Income', fontweight = 'bold')
plt.tight_layout()
plt.show()

plt.figure(figsize = (20, 7))
plt.title('Agglomerative Clustering (Linkage = Ward)', size = 18, fontweight = 'bold')
hierarchy.dendrogram(hierarchy.linkage(income, method = 'ward'), labels = data['country'].tolist())
plt.xlabel('Country', fontweight = 'bold')
plt.ylabel('Income', fontweight = 'bold')
plt.tight_layout()
plt.show()

plt.title('Silhouette', size = 18, fontweight = 'bold')
plt.scatter(range(2, 11), silhouette)
plt.plot(range(2, 11), silhouette)
plt.xlabel('k', fontweight = 'bold')
plt.ylabel('Quality of clusters', fontweight = 'bold')
plt.show()