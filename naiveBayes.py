import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

f = 'processed.cleveland.data'

col_names = ['age', 'sex', 'chest_pain', 'b_pressure', 'cholesterol', \
        'b_sugar_up', 'ecg_type', 'heart_rate', 'exer_angina', \
        'exer_depress', 'exer_slope', 'fluor_count', 'thal_defect', 'risk']
cleveland = pd.read_csv(f, names = col_names)
for c in col_names:
    cleveland[c] = cleveland[c].apply(lambda s: np.nan if s=='?' else float(s))
    # '?' char is nan, columns with it are str, convert to numeric
cleveland = cleveland.dropna()
cleveland.x = cleveland.loc[:,'age':'thal_defect']
cleveland['Y'] = cleveland['risk'].apply(lambda x: 1 if x>0 else 0)

gnb = GaussianNB()
y_pred = gnb.fit(cleveland.x, cleveland['Y']).predict(cleveland.x)
print("Number of mislabeled points out of a total %d points : %d" % (cleveland.x.shape[0],(cleveland['Y'] != y_pred).sum()))
print("Accuracy: %f" % (1-(float((cleveland['Y'] != y_pred).sum())/float(cleveland.x.shape[0]))))
