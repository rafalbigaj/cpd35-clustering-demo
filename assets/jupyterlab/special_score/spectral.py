from sklearn import cluster


def spectral():
    return cluster.SpectralClustering(
        n_clusters=2,
        assign_labels='discretize',
        random_state=0)

