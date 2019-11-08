import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pandas.read_csv('NewКластеризация.csv', sep=";", index_col="Регион")
print(df)

kmeans = KMeans(n_clusters=5).fit(df)
centroids = kmeans.cluster_centers_

plt.scatter(df["Среднемесячный размер социальной поддержки на одного пользователя"], df["Количество преступлений на 100000 человек"], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

pl1 = df.plot.scatter(x="Количество преступлений на 100000 человек", y="Общая площадь жилых помещений")
pl2 = df.plot.scatter(x="Среднемесячный размер социальной поддержки на одного пользователя", y="Общая площадь жилых помещений")
pl3 = df.plot.scatter(x="Среднемесячный размер социальной поддержки на одного пользователя", y="Количество преступлений на 100000 человек")
plt.show()


