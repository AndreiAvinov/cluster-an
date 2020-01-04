import pandas
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# import dataframe
df = pandas.read_csv('NewКластеризация.csv', sep=";", index_col="Регион")
print(df)
N_COLUMNS = len(df.columns)
N_ROWS = len(df.index)

# Normalizing dataframe
for i in range(N_COLUMNS):
    df.iloc[:, i] = ((df.iloc[:, i] - df.iloc[:, i].min()) / (df.iloc[:, i].max() - df.iloc[:, i].min()))

# KMeans simple
kmeans = KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_

# Hierarchy
dendrogram = sch.dendrogram(sch.linkage(df, method='ward'), labels=df.index)
ac = AgglomerativeClustering(n_clusters=3).fit(df)

# pl = plt.scatter(df.iloc[:, 3], df["Количество преступлений на 100000 человек"], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
# pl1 = df.plot.scatter(x="Количество преступлений на 100000 человек", y="Общая площадь жилых помещений")
# pl2 = df.plot.scatter(x="Среднемесячный размер социальной поддержки на одного пользователя", y="Общая площадь жилых помещений")
# pl3 = df.plot.scatter(x="Среднемесячный размер социальной поддержки на одного пользователя", y="Количество преступлений на 100000 человек")

df["KMeans"] = kmeans.labels_
df["AC"] = ac.labels_

with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.sort_values("AC"))
plt.show()

df.to_csv(path_or_buf="res.csv")
