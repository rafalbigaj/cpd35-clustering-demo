from sklearn.pipeline import Pipeline
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn import cluster
from preprocess_data import one_hot_encoder


def spectral(assign_labels='discretize'):
    return cluster.SpectralClustering(
        n_clusters=2,
        assign_labels=assign_labels,
        random_state=0)

def dbscan():
    return cluster.DBSCAN(eps=2.0, min_samples=2)

def kmeans():
    return cluster.KMeans(n_clusters=2, random_state=0)


spectral_discretize = spectral('discretize')
spectral_kmeans = spectral('kmeans')
db_scan = dbscan()
kmeans = kmeans()

clusterings = [
    ['spectral (discretize)', spectral_discretize],
    ['spectral (kmeans)', spectral_kmeans],
    ['DBSCAN', db_scan],
    ['Kmeans', kmeans],
]


scores = [
    ['homogeneity', homogeneity_score],
    ['completeness', completeness_score],
    ['v_measure', v_measure_score],
]


def train(input_df, clustering_name, clustering_op):
    transformer = one_hot_encoder(input_df)
    pipeline_linear = Pipeline([('transformer', transformer), (clustering_name, clustering_op)])
    model = pipeline_linear.fit(input_df)
    return model


def evaluate(reference_df, clustering_op):
    labels_true = [int(l == 'Risk') for l in reference_df['Risk']]
    labels_pred = clustering_op.labels_
    result = {}
    
    for (score_name, score_fn) in scores:
        score_value = score_fn(labels_true, labels_pred)
        print("{}: {}".format(score_name, score_value))
        result[score_name] = score_value
    
    return result