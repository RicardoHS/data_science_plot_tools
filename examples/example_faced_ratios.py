import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

import sys
sys.path.append('..')

from analysis import features

def main():
    iris = datasets.load_iris()
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                        columns= iris['feature_names'] + ['target'])

    f,a = features.faced_ratios(df, df.columns[:-1].values, 'target', n_rows=2, n_cols=2)
    plt.show()

main()
