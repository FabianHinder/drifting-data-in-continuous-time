# -*- coding: utf-8 -*-
import numpy as np
import time


# Rotating hyperplane dataset
def create_hyperplane_dataset(n_samples, n_dim=2, plane_angle=0.45):
    w = np.dot(np.array([[np.cos(plane_angle), -np.sin(plane_angle)], [np.sin(plane_angle), np.cos(plane_angle)]]), np.array([1.0, 1.0]))
    X = np.random.uniform(-1.0, 1.0, (n_samples, n_dim))
    Y = np.array([1 if np.dot(x, w) >= 0 else 0 for x in X])
    
    return X, Y


def create_rotating_hyperplane_dataset(n_samples_per_concept=200, concepts=np.arange(0.0, 5.0, 1.0)):
    X_stream = []
    Y_stream = []
    concept_drifts = []
    
    t = 0
    for a in concepts:
        data_stream_X, data_stream_Y = create_hyperplane_dataset(n_samples=n_samples_per_concept, plane_angle=a)
        t += n_samples_per_concept

        X_stream.append(data_stream_X)
        Y_stream.append(data_stream_Y)
        concept_drifts.append(t)
    concept_drifts.pop()

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)), "drifts": np.array(concept_drifts)}


# SEA
def sea_clf_0(x):
    return 0 if x[0] + x[1] <= 8 else 1

def sea_clf_1(x):
    return 0 if x[0] + x[1] <= 9 else 1

def sea_clf_2(x):
    return 0 if x[0] + x[1] <= 7 else 1

def sea_clf_3(x):
    return 0 if x[0] + x[1] <= 9.5 else 1


def create_sea_dataset(n_samples, clf_id=0):
    clf = None
    if clf_id == 0:
        clf = sea_clf_0
    elif clf_id == 1:
        clf = sea_clf_1
    elif clf_id == 2:
        clf = sea_clf_2
    elif clf_id == 3:
        clf = sea_clf_3

    X = (11 - 0) * np.random.random((n_samples, 2)) + 0
    Y = np.apply_along_axis(clf, axis=1, arr=X)

    return X, Y


def create_sea_drift_dataset(n_samples_per_concept=200, concepts=[0, 1, 2, 3]):
    X_stream = []
    Y_stream = []
    concept_drifts = []
    
    t = 0
    for c in concepts:
        data_stream_X, data_stream_Y = create_sea_dataset(n_samples=n_samples_per_concept, clf_id=c)
        t += n_samples_per_concept

        X_stream.append(data_stream_X)
        Y_stream.append(data_stream_Y)
        concept_drifts.append(t)
    concept_drifts.pop()

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)), "drifts": np.array(concept_drifts)}


# Drift detection
class DriftDetectorSupervised():
    def __init__(self, clf, drift_detector, training_buffer_size):
        self.clf = clf
        self.drift_detector = drift_detector
        self.training_buffer_size = training_buffer_size
        self.X_training_buffer = []
        self.Y_training_buffer = []
        self.changes_detected = []

    def apply_to_stream(self, X_stream, Y_stream):
        self.changes_detected = []

        collect_samples = False
        T = len(X_stream)
        for t in range(T):
            x, y = X_stream [t,:], Y_stream[t,:]
            
            if collect_samples == False:
                self.drift_detector.add_element(self.clf.score(x, y))

                if self.drift_detector.detected_change():
                    self.changes_detected.append(t)
                    
                    collect_samples = True
                    self.X_training_buffer = []
                    self.Y_training_buffer = []
            else:
                self.X_training_buffer.append(x)
                self.Y_training_buffer.append(y)

                if len(self.X_training_buffer) >= self.training_buffer_size:
                    collect_samples = False
                    self.clf.fit(np.array(self.X_training_buffer), np.array(self.Y_training_buffer))
        
        return self.changes_detected


class DriftDetectorUnsupervised():
    def __init__(self, drift_detector, batch_size):
        self.drift_detector = drift_detector
        self.batch_size = batch_size
        self.changes_detected = []

    def apply_to_stream(self, data_stream):
        self.changes_detected = []
        n_data_stream_samples = len(data_stream)

        t = 0
        while t < n_data_stream_samples:
            end_idx = t+self.batch_size
            if end_idx >= n_data_stream_samples:
                end_idx = n_data_stream_samples

            batch = data_stream[t:end_idx, :]
            self.drift_detector.add_batch(batch)

            if self.drift_detector.detected_change():
                self.changes_detected.append(t)
            
            t += self.batch_size
        
        return self.changes_detected


# Evaluation
def evaluate(true_concept_drifts, pred_concept_drifts, tol=50):
    false_alarms = 0
    drift_detected = 0
    drift_not_detected = 0

    # Check for false alarms
    for t in pred_concept_drifts:
        b = False
        for dt in true_concept_drifts:
            if dt <= t and t <= dt + tol:
                b = True
                break
        if b is False:  # False alarm
            false_alarms += 1
    
    # Check for detected and undetected drifts
    for dt in true_concept_drifts:
        b = False
        for t in pred_concept_drifts:
            if dt <= t and t <= dt + tol:
                b = True
                drift_detected += 1
                break
        if b is False:
            drift_not_detected += 1

    #return {"false_alarms": false_alarms, "drift_detected": drift_detected, "drift_not_detected": drift_not_detected}
    
    # Compute F1-score
    tp = drift_detected
    fp = false_alarms
    fn = drift_not_detected
    if tp == fp and tp == 0:
        tp = 1e-10
        fp = 1e-10
    if tp == fn and tp == 0:
        tp = 1e-10
        fp = 1e-10

    p = tp / (1.*tp + fp)
    r = tp / (1.*tp + fn)
    if p+r == 0:
        r += 1.e-10
    return 2. * ((p*r) / (p+r))


# Classifier
from sklearn.svm import SVC
class Classifier():
    def __init__(self):
        self.model = SVC(C=1.0, kernel='linear')
        self.flip_score = False
    
    def fit(self, X, y):
        self.model.fit(X, y.ravel())
    
    def score(self, x, y):
        s = int(self.model.predict([x]) == y)
        if self.flip_score == True:
            return 1 - s
        else:
            return s
    
    def score_set(self, X, y):
        return self.model.score(X, y.ravel())


if __name__ == "__main__":
    dataset_rotplane = False

    # Create data set
    D = create_rotating_hyperplane_dataset(n_samples_per_concept=200, concepts=[0.0, 2.0, 4.0])
    if dataset_rotplane is False:
        D = create_sea_drift_dataset(n_samples_per_concept=200, concepts=[0, 1, 2])
    
    concept_drifts = D["drifts"]
    X, Y = D["data"]
    data_stream = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
    print(concept_drifts)

    X0, Y0 = X[0:200, :], Y[0:200, :]   # Training dataset
    data0 = data_stream[0:200,:]

    # Run drift detector
    from HDDDM import HDDDM
    from SWIDD import SWIDD
    from skmultiflow.drift_detection import ADWIN, DDM, EDDM
    
    # Unsupervised
    drift_detectors = [("SWIDD", DriftDetectorUnsupervised(SWIDD(max_window_size=100, min_window_size=70), batch_size=1)),
                        ("HDDDM", DriftDetectorUnsupervised(HDDDM(data0, gamma=None, alpha=0.005), batch_size=50)),
                        ("K2ST", DriftDetectorUnsupervised(HDDDM(data0, gamma=None, alpha=0.05, use_k2s_test=True), batch_size=50)),
                        ("MMD2", DriftDetectorUnsupervised(HDDDM(data0, gamma=None, alpha=0.005, use_mmd2=True), batch_size=50))]
    for desc, dd in drift_detectors:
        print("***********")
        print(desc)
        print("**********")

        s = time.time()
        changes_detected = dd.apply_to_stream(data_stream)
        t = time.time() - s

        print("Time (s): {0}s".format(t))

        print("Score: {0}".format(evaluate(concept_drifts, changes_detected, tol=100)))
        print()

    #from plotting import plot_classification_dataset
    #plot_classification_dataset(X0, Y0.ravel())

    # Supervised
    drift_detectors = [("ADWIN", ADWIN()), ("DDM", DDM(min_num_instances=30, warning_level=2.0, out_control_level=3.0)), ("EDDM", EDDM())]
    training_buffer_size = 50
    if dataset_rotplane is False:
        drift_detectors = [("ADWIN", ADWIN(delta=2.)), ("DDM", DDM(min_num_instances=30, warning_level=2.0, out_control_level=3.0)), ("EDDM", EDDM())]
        training_buffer_size = 100

    for desc, dd in drift_detectors:
        print("***********")
        print(desc)
        print("**********")
        
        clf = Classifier()
        clf.fit(X0, Y0)

        if not isinstance(dd, ADWIN):
            clf.flip_score = True
        dd = DriftDetectorSupervised(clf=clf, drift_detector=dd, training_buffer_size=training_buffer_size)

        s = time.time()
        changes_detected = dd.apply_to_stream(X, Y)
        t = time.time() - s

        print("Time (s): {0}s".format(t))

        print(evaluate(concept_drifts, changes_detected, tol=100))
        print()
