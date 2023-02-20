import pandas as pd
import numpy as np

A = pd.read_csv("ground_truth.csv", header=None).to_numpy()
A = np.abs(A)
W = np.maximum(A, A.T)
np.savetxt("ground_truth_skel.csv", W, delimiter=",")
