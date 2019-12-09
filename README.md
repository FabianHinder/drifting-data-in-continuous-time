# A probability theoretic approach to drifting data in continuous time domains

This repository contains the implementation of the methods proposed in the paper [A probability theoretic approach to drifting data in continuous time domains](Paper.pdf) by Fabian Hinder, André Artelt and Barbara Hammer.
- The *Single-Window-Independence-Drift-Detector (SWIDD)* is implemented in [SWIDD.py](SWIDD.py). If your want to use a different/custom test for independence, you have to overwrite the method `_test_for_drift`.
- The Hellinger-Distance-Drift-Detection-Method (HDDDM) is implemented in [HDDDM.py](HDDDM.py).
- The experiments for comparing different drift detection methods are implemented in [experiments_drift_detectors.py](experiments_drift_detectors.py).
- [The Least-Squares-Independence-Test (LSIT)](http://www.ms.k.u-tokyo.ac.jp/2011/LSIT.pdf) is implemented in [lsit.py](lsit.py).
- The experiments on the toy data sets are implemented in [experiment_hdddm.py](experiment_hdddm.py) and [experiment_adwin.py](experiment_adwin.py).
- The *k-curve-DriFDA* is implemented in [k_curve_DriFDA.py](k_curve_DriFDA.py). A toy example is implemented in [k_curve_example.py](k_curve_example.py).
- The *linear-DriFDA* is implemented in [linear_DriFDA.py](linear_DriFDA.py). A toy example is implemented in [linear_DriFDA_example.py](linear_DriFDA_example.py).

## Requirements

- Python >= 3.6
- Packages as listed in [REQUIREMENTS.txt](REQUIREMENTS.txt)

## Third party components

- [kernel_two_sample_test.py](https://github.com/emanuele/kernel_two_sample_test/blob/master/kernel_two_sample_test.py) is taken from [GitHub](https://github.com/emanuele/kernel_two_sample_test) and implements the kernel two-sample tests as in Gretton et al 2012 (JMLR).
- [HSIC.py](https://raw.githubusercontent.com/amber0309/HSIC/master/HSIC.py) is taken from [GitHub](https://github.com/amber0309/HSIC) and implements the [Hilbert-Schmidt Independence Criterion (HSIC)](http://papers.nips.cc/paper/3201-a-kernel-statistical-test-of-independence.pdf).
- [mutual_info.py](mutual_info.py) is taken from [GitHub](https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429) and contains an implementation for a non-parametric computation of entropy and mutual-information.


## How to cite

You can cite the version on [arXiv](https://arxiv.org/abs/1912.01969).
