from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd

data = pd.read_csv("Country-data.csv")

income = data[['income']]

sse = []
silhouette = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, n_init = 'auto', random_state = 42)
    sse.append(kmeans.fit(income).inertia_)
    if i > 1:
        silhouette.append(silhouette_score(income, kmeans.fit_predict(income)))

kmeans = KMeans(n_clusters = 2, n_init = 'auto', random_state = 42)
kmeans_ = kmeans.fit_predict(income)
data['cluster'] = kmeans_

plt.figure(figsize = (20, 7))
plt.title('K-means Clustering', size = 18, fontweight = 'bold')
plt.scatter(data['country'], data['income'], c = data['cluster'], cmap = 'plasma')
plt.xticks(rotation = 90)
plt.xlabel('Country', fontweight = 'bold')
plt.ylabel('Income', fontweight = 'bold')
plt.tight_layout()
plt.show()

plt.title('Elbow Method', size = 18, fontweight = 'bold')
plt.scatter(range(1, 11), sse)
plt.plot(range(1, 11), sse)
plt.xlabel('k', fontweight = 'bold')
plt.ylabel('SSE', fontweight = 'bold')
plt.show()

plt.title('Silhouette', size = 18, fontweight = 'bold')
plt.scatter(range(2, 11), silhouette)
plt.plot(range(2, 11), silhouette)
plt.xlabel('k', fontweight = 'bold')
plt.ylabel('Quality of clusters', fontweight = 'bold')
plt.show()