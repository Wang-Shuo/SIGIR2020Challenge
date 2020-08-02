import pandas as pd
import numpy as np



train = pd.read_csv('/home/competition/SIGIR2020/Task_1/data/preprocessed/train.tsv', sep = '\t')
pyx = np.zeros([train.shape[0], 27])

idx = 0 
for i in range(4):
    prob_df = pd.read_csv('results/cv{}_prob_result.tsv'.format(i), sep = '\t')
    pyx[idx:(idx + prob_df.shape[0])] = prob_df[['class_{}_prob'.format(i) for i in range(27)]].values
    idx += prob_df.shape[0]

np.save('results/psx.npy', pyx)
    
