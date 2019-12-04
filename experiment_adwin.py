# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import savemat
from HDDDM import HDDDM
from plotting import plot_classification_dataset, plot_2d_decisionboundary
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle


def create_hyperplane_dataset(n_samples, n_dim=2, plane_angle=0.45):
    w = np.dot(np.array([[np.cos(plane_angle), -np.sin(plane_angle)], [np.sin(plane_angle), np.cos(plane_angle)]]), np.array([1.0, 1.0]))
    X = np.random.uniform(-1.0, 1.0, (n_samples, n_dim))
    Y = np.array([1 if np.dot(x, w) >= 0 else 0 for x in X])
    
    return X, Y


class Classifier():
    def __init__(self):
        self.model = SVC(C=1.0, kernel='linear')
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def score(self, x, y):
        return int(self.model.predict([x]) == y)
    
    def score_set(self, X, y):
        return self.model.score(X, y)


#n_wrong_samples = 10
#n_correct_samples = 90
n_wrong_samples = 60
n_correct_samples = 120

def create_training_dataset():
    np.random.seed(42)

    X = []
    Y = []

    centerA = np.array([[-2., 2.]])
    centerB = np.array([[2., -2.]])

    # Class A
    x, _ = make_blobs(n_samples=n_wrong_samples, n_features=2, centers=centerA, cluster_std=0.5)
    X.append(x)
    Y.append([1 for _ in range(n_wrong_samples)])
    
    x, _ = make_blobs(n_samples=n_correct_samples, n_features=2, centers=centerA, cluster_std=0.8)
    X.append(x)
    Y.append([0 for _ in range(n_correct_samples)])

    # Class B
    x, _ = make_blobs(n_samples=n_wrong_samples, n_features=2, centers=centerB, cluster_std=0.5)
    X.append(x)
    Y.append([0 for _ in range(n_wrong_samples)])
    
    x, _ = make_blobs(n_samples=n_correct_samples, n_features=2, centers=centerB, cluster_std=0.8)
    X.append(x)
    Y.append([1 for _ in range(n_correct_samples)])

    return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)

def create_test_dataset():
    np.random.seed(42)

    X = []
    Y = []

    centerA = np.array([[-2., 2.]])
    centerA2 = np.array([[2., 5.0]])
    centerB = np.array([[2., -2.]])
    centerB2 = np.array([[5.5, 2.]])

    # Class A
    x, _ = make_blobs(n_samples=n_wrong_samples, n_features=2, centers=centerA2, cluster_std=0.5)
    X.append(x)
    Y.append([1 for _ in range(n_wrong_samples)])
    
    x, _ = make_blobs(n_samples=n_correct_samples, n_features=2, centers=centerA, cluster_std=0.8)
    X.append(x)
    Y.append([0 for _ in range(n_correct_samples)])

    # Class B
    x, _ = make_blobs(n_samples=n_wrong_samples, n_features=2, centers=centerB, cluster_std=0.5)
    X.append(x)
    Y.append([0 for _ in range(n_wrong_samples)])
    
    x, _ = make_blobs(n_samples=n_correct_samples, n_features=2, centers=centerB2, cluster_std=0.8)
    X.append(x)
    Y.append([1 for _ in range(n_correct_samples)])

    return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)


if __name__ == "__main__":
    np.random.seed(2)

    # Support vector machine classifier
    clf = Classifier()

    X, Y = create_training_dataset()
    X, Y = shuffle(X, Y)
    #X /= np.std(X)
    print(X.shape)
    print(Y.shape)
    #plot_classification_dataset(X, Y)

    clf.fit(X, Y)

    print(clf.score_set(X, Y))
    plot_2d_decisionboundary(clf.model, X, Y, show=True, savefig_path="../tex/imgs/data0.png")

    # Drift!
    X2, Y2 = create_test_dataset()
    X2, Y2 = shuffle(X2, Y2)
    print(clf.score_set(X2, Y2))
    plot_2d_decisionboundary(clf.model, X2, Y2, show=True, savefig_path="../tex/imgs/data0drifted.png")

    # Construct final data stream
    data_stream_X = np.concatenate((X, X2, X, X2), axis=0)
    data_stream_Y = np.concatenate((Y, Y2, Y, Y2), axis=0).reshape(-1, 1)
    #data_stream_X /= np.std(data_stream_X)
    print(data_stream_X.shape)

    # Export to .mat
    #X_raw = data_stream_X
    #Y_raw = data_stream_Y
    #savemat("data_stream_ADWIN.mat", {'X0_raw': X.T, 'Y0_raw': Y.reshape(1, -1).astype(np.float64), 'X1_raw': X2.T, 'Y1_raw': Y2.reshape(1, -1).astype(np.float64), 'X_stream_raw': data_stream_X.T, 'Y_stream_raw': data_stream_Y.reshape(1, -1).astype(np.float64)})

    # Run ADWIN
    from skmultiflow.drift_detection.adwin import ADWIN
    adwin = ADWIN()

    for i in range(len(data_stream_X)):
        y = clf.score(data_stream_X[i, :], data_stream_Y[i, :])
        adwin.add_element(y)

        if adwin.detected_change():
            print('Change detected in data at index: ' + str(i))
