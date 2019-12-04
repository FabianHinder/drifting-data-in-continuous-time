# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import savemat
from HDDDM import HDDDM
from plotting import plot_classification_dataset, plot_2d_decisionboundary
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle



class Classifier():
    def __init__(self):
        self.model = QuadraticDiscriminantAnalysis()
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def score(self, x, y):
        return int(self.model.predict([x]) == y)
    
    def score_set(self, X, y):
        return self.model.score(X, y)


n_samples = 100

def create_training_dataset():
    np.random.seed(42)

    #X0 = (np.random.rand(n_samples, 1) * 2. - 1.) * 20.
    X0 = np.concatenate((np.arange(0.4, 20.4, step=20. / 50.), np.arange(-20., 0., step=20. / 50.)), axis=0).reshape(-1, 1)
    X0 = np.concatenate([X0, -1. * X0], axis=1)
    Y0 = np.array([0 for _ in range(n_samples)])

    #X1 = (np.random.rand(n_samples, 1) * 2. - 1.) * 20.
    X1 = np.concatenate((np.arange(0.4, 20.4, step=20. / 50.), np.arange(-20., 0., step=20. / 50.)), axis=0).reshape(-1, 1)
    X1 = np.concatenate([X1, 1. * X1], axis=1)
    Y1 = np.array([1 for _ in range(n_samples)])

    return np.concatenate([X0, X1], axis=0), np.concatenate([Y0, Y1], axis=0)

def create_test_dataset():
    np.random.seed(42)

    #X0 = (np.random.rand(n_samples, 1) * 2. - 1.) * 20.
    X0 = np.concatenate((np.arange(0.4, 20.4, step=20. / 50.), np.arange(-20., 0., step=20. / 50.)), axis=0).reshape(-1, 1)
    X0 = np.concatenate([X0, -1. * X0], axis=1)
    Y0 = np.array([1 for _ in range(n_samples)])

    #X1 = (np.random.rand(n_samples, 1) * 2. - 1.) * 20.
    X1 = np.concatenate((np.arange(0.4, 20.4, step=20. / 50.), np.arange(-20., 0., step=20. / 50.)), axis=0).reshape(-1, 1)
    X1 = np.concatenate([X1, 1. * X1], axis=1)
    Y1 = np.array([0 for _ in range(n_samples)])

    return np.concatenate([X0, X1], axis=0), np.concatenate([Y0, Y1], axis=0)


if __name__ == "__main__":
    np.random.seed(2)

    # QDA classifier
    clf = Classifier()

    X, Y = create_training_dataset()
    X, Y = shuffle(X, Y)
    #X /= np.std(X)
    print(X.shape)
    print(Y.shape)
    #plot_classification_dataset(X, Y)

    clf.fit(X, Y)

    print(clf.score_set(X, Y))
    plot_2d_decisionboundary(clf.model, X, Y, show=True, savefig_path="../tex/imgs/data1.png")

    # Drift!
    X2, Y2 = create_test_dataset()
    X2, Y2 = shuffle(X2, Y2)
    #X2 /= np.std(X2)
    print(clf.score_set(X2, Y2))
    plot_2d_decisionboundary(clf.model, X2, Y2, show=True, savefig_path="../tex/imgs/data1drifted.png")

    # Construct final data stream
    data_stream_X = np.concatenate((X, X2), axis=0)
    data_stream_Y = np.concatenate((Y, Y2), axis=0).reshape(-1, 1)
    data_stream_X /= np.std(data_stream_X)  # Somehow important for the test
    print(data_stream_X.shape)

    # Export to .mat
    X_raw = data_stream_X
    Y_raw = data_stream_Y
    #savemat("../../data_stream_HDDDM.mat", {'X0_raw': X.T, 'Y0_raw': Y.reshape(1, -1).astype(np.float64), 'X1_raw': X2.T, 'Y1_raw': Y2.reshape(1, -1).astype(np.float64), 'X_stream_raw': data_stream_X.T, 'Y_stream_raw': data_stream_Y.reshape(1, -1).astype(np.float64)})

    # Run HDDDM
    data0 = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
    data1 = np.concatenate((X2, Y2.reshape(-1, 1)), axis=1)
    data_stream = np.concatenate((data0, data1), axis=0)
    n_data_stream_samples = data_stream.shape[0]

    # Inspect histograms
    from HDDDM import compute_histogram
    hist0 = compute_histogram(data0, data0.shape[0])
    hist1 = compute_histogram(data1, data0.shape[0])

    print("Histograms are preserved: {0}".format(np.all(hist0 == hist1)))   # Histograms are equal. Thus, HDDDM can not detect the drift!

    # Inspect mean and variance
    print("Means are preserved: {0}".format(np.mean(data0, axis=0) == np.mean(data1, axis=0))) # Mean and variance are equal
    print("Variances are preserved: {0}".format(np.var(data0, axis=0) == np.var(data1, axis=0)))


    # Class wise mean and variance before and after the drift: Both are equal! Thus, testing for a change in distribution by considering the mean and variance only will fail!
    print("Means of class 0 are preserved: {0}".format(np.around(np.mean(X[Y == 0,:], axis=0), decimals=10) == np.around(np.mean(X2[Y2 == 0,:], axis=0), decimals=10)))
    print("Variances of class 0 are preserved: {0}".format(np.var(X[Y == 0,:], axis=0) == np.var(X2[Y2 == 0,:], axis=0)))

    print("Means of class 1 are preserved: {0}".format(np.around(np.mean(X[Y == 1,:], axis=0), decimals=10) == np.around(np.mean(X2[Y2 == 1,:], axis=0), decimals=10)))
    print("Variances of class 1 are preserved: {0}".format(np.var(X[Y == 1,:], axis=0) == np.var(X2[Y2 == 1,:], axis=0)))

    ###################################################################################################################################################################################

    print("Running HDDDM...")

    batch_size = 50

    hdddm = HDDDM(data0, gamma=None, alpha=0.005)
    
    i = 0
    while i < n_data_stream_samples:
        end_idx = i+batch_size
        if end_idx >= n_data_stream_samples:
            end_idx = n_data_stream_samples

        # Add batch
        batch = data_stream[i:end_idx, :]
        hdddm.add_batch(batch)
        
        # Drift detected?
        if hdddm.detected_change():
            print('Change detected in data at index: {0} - {1}'.format(i, end_idx))
        
        i += batch_size

    # Run SWIDD
    print("Running SWIDD...")

    from SWIDD import SWIDD
    swidd = SWIDD(max_window_size=200, min_window_size=100, min_p_value=0.1)

    for i in range(len(data_stream)):
        swidd.add_batch(data_stream[i, :])

        if swidd.detected_change():
            print('Change detected in data at index: ' + str(i))
