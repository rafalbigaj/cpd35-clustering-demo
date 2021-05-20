from sklearn import cluster


def my_dbscan():
    return cluster.DBSCAN(eps=2.0, min_samples=2)
