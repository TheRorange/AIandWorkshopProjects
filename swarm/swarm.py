import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


class DataBot:
    def __init__(self, position, data_point):
        self.position = position
        self.data_point = data_point

    def to_polar(self):
        x, y = self.position
        radius = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)
        return radius, angle

def initialize_databots(data):
    mds = MDS(n_components=2, random_state=0)
    reduced_data = mds.fit_transform(data)
    databots = [DataBot(position, data_point) for position, data_point in zip(reduced_data, data)]
    return databots

def move_databots(databots, iterations=10):
    for _ in range(iterations):
        for db in databots:
            db.position += np.random.normal(0, 0.1, 2)
    return databots

def perform_clustering(databots):
    polar_positions = np.array([db.to_polar() for db in databots])
    kmeans = KMeans(n_clusters=3, random_state=0)  # Assuming 3 clusters
    kmeans.fit(polar_positions)
    return kmeans.labels_

def simplified_pswarm_clustering(data):
    databots = initialize_databots(data)
    databots = move_databots(databots)
    cluster_labels = perform_clustering(databots)
    return cluster_labels

def project_high_dimensional_data(data):
    distance_matrix = pairwise_distances(data, metric='euclidean')
    mds = MDS(n_components=2, dissimilarity='precomputed')
    reduced_data = mds.fit_transform(distance_matrix)
    return reduced_data

def calculate_generalized_u_matrix(reduced_data):
    distances = pairwise_distances(reduced_data, metric='euclidean')
    u_matrix = distances.reshape((int(np.sqrt(len(distances))), -1))
    return u_matrix

def perform_clustering(reduced_data):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(reduced_data)
    return kmeans.labels_

def calculate_accuracy(labels_true, labels_pred):
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    true_positives = count_true_positives(labels_true, labels_pred)
    accuracy = true_positives / len(labels_true)
    return accuracy

def count_true_positives(labels_true, labels_pred):
    size = max(max(labels_true), max(labels_pred)) + 1
    confusion_matrix = np.zeros((size, size), dtype=int)
    for true_label, pred_label in zip(labels_true, labels_pred):
        confusion_matrix[true_label][pred_label] += 1

    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    
    true_positives = confusion_matrix[row_ind, col_ind].sum()
    return true_positives


def main():
    data = pd.read_csv('hepta.csv')
    reduced_data = project_high_dimensional_data(data)
    u_matrix = calculate_generalized_u_matrix(reduced_data)
    cluster_labels = perform_clustering(reduced_data)
    accuracy = calculate_accuracy(cluster_labels)
    print(reduced_data)

if __name__ == "__main__":
    main()
