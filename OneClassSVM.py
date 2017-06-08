import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
# from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

colors = ['m', 'g', 'b']
legend1 = {}


df = pd.read_csv("testFitBit.csv", parse_dates=['time'], index_col='time')
df.index = pd.to_datetime(df.index, format='%H:%M:%S')
df.drop(df.columns[[0]], axis=1, inplace=True)
length = 96

timeseries = df


# rng = np.random.RandomState(42)
# clf = IsolationForest(max_samples=96, random_state=rng)
clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)


clf.fit(timeseries)
# y_pred_train = clf.predict(timeseries)
# y_pred_test = clf.predict([1000])
#
# print y_pred_train
# Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
# Z1 = Z1.reshape(xx1.shape)
# Z1.plot()
z = clf.decision_function(timeseries)
print z
plt.plot(z)
# legend1["SVM"] = plt.contour(xx1, yy1, Z1, levels=[0], linewidths=2, colors='m')
#
# legend1_values_list = list(legend1.values())
# legend1_keys_list = list(legend1.keys())
#
# plt.title("Outlier detection")
# plt.scatter(X1[:, 0], X1[:, 1], color='black')
# bbox_args = dict(boxstyle="round", fc="0.8")
# arrow_args = dict(arrowstyle="->")
#
# plt.xlim((xx1.min(), xx1.max()))
# plt.ylim((yy1.min(), yy1.max()))
#
plt.show()
