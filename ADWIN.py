# -*- coding: utf-8 -*-
import numpy as np
from lsit import LSIT, test_independence


class MyAdwin():
    def __init__(self, y_type=0, max_window_size=100, min_window_size=70, min_p_value=0.1):
        self.y_type = y_type
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.min_p_value = min_p_value

        self.X = []
        self.Y = []
        self.n_items = 0
        self.min_n_items = self.min_window_size / 4.

        self.drift_detected = False
    
    def __test_for_drift(self):
        t = np.array(range(self.n_items)) / (1. * self.n_items)
        #t -= np.mean(t)
        t /= np.std(t)
        t = t.reshape(-1, 1)

        #return LSIT(np.hstack((np.array(self.X), np.array(self.Y))), t, self.y_type, T=10, b=10, fold=3, verbose=False, n_jobs=-1)["pvalue"]
        return 1.0 if test_independence(np.hstack((np.array(self.X), np.array(self.Y))), t) == True else 0.0
        #return LSIT(np.array(self.X), np.array(self.Y), self.y_type, T=10, b=10, fold=3, verbose=False, n_jobs=-1)["pvalue"]

    def add_element(self, x, y):
        self.drift_detected = False

        # Add item
        self.X.append(x)
        self.Y.append(y)
        self.n_items += 1
        
        # Is buffer full?
        if self.n_items > self.max_window_size:
            self.X.pop(0)
            self.Y.pop(0)
            self.n_items -= 1

        # Enough items for testing for drift?
        if self.n_items >= self.min_window_size:
            # Test for drift
            p = self.__test_for_drift()
            #print("p-value: {0}".format(p))

            if p <= self.min_p_value:
            #if p >= self.min_p_value:
                self.drift_detected = True
                
                # Remove samples until no drift is present!
                while p <= self.min_p_value and self.n_items >= self.min_n_items:
                #while p >= self.min_p_value and self.n_items >= self.min_n_items:
                    #print("Remove sample")
                    # Remove old samples
                    self.X.pop(0)
                    self.Y.pop(0)
                    self.n_items -= 1

                    # Check for independence <=> Check for drift
                    p = self.__test_for_drift()


    def detected_change(self):
        return self.drift_detected


if __name__ == "__main__":
    # Create new model
    ##adwin = MyAdwin(y_type=0, max_window_size=100, min_window_size=80, min_p_value=0.1)
    adwin = MyAdwin(y_type=1, max_window_size=200, min_window_size=200, min_p_value=0.1)
    #from skmultiflow.drift_detection.adwin import ADWIN
    #adwin = ADWIN()

    """
    # Create two regression data sets and join them (concept drift!)
    n = 100

    X1 = (np.random.rand(n, 1) * 2. - 1.) * 20.
    Y1 = -0.5 * X1 + 0.5
    X2 = (np.random.rand(n, 1) * 2. - 1.) * 20.
    Y2 = 0.5 * X2 + 0.5 #np.random.randn(n, 1) + np.sin(X2 / 20.*np.pi)

    data_stream_X = np.concatenate((X1, X2), axis=0)
    data_stream_Y = np.concatenate((Y1, Y2), axis=0)

    data_stream_X /= np.std(data_stream_X)
    data_stream_Y /= np.std(data_stream_Y)
    """
    # Create a rotating hyperplane data set
    from demo import create_hyperplane_dataset
    X = []
    Y = []
    for a in [0.0, 2.0, 4.0]:#np.arange(0.0, 5.0, 1.0):
        data_stream_X, data_stream_Y = create_hyperplane_dataset(n_samples=400, plane_angle=a)
        X.append(data_stream_X)
        Y.append(data_stream_Y)
    
    data_stream_X = np.concatenate(X, axis=0)
    data_stream_Y = np.concatenate(Y, axis=0).reshape(-1, 1)

    data_stream_X /= np.std(data_stream_X)
    #data_stream_Y /= np.std(data_stream_Y)
    print(data_stream_X.shape)
    print(data_stream_Y.shape)

    #"""
    # Adding stream elements to ADWIN and check for drift
    for i in range(len(data_stream_X)):
        #print("{}/{}".format(i+1, len(data_stream_X)))
        adwin.add_element(data_stream_X[i, :], data_stream_Y[i, :])
        
        if adwin.detected_change():
            print('Change detected in data at index: ' + str(i))
    #"""
