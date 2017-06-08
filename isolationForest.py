import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import datetime
from sklearn.ensemble import IsolationForest

def plotTS(timeseries):
    rolmean = timeseries.rolling(window=12,center=False).mean()
    rolstd = timeseries.rolling(window=12,center=False).std()
    #
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

def plotIsolationForest(timeseries):
    xx, yy = np.meshgrid(np.linspace(0, 24, 50), np.linspace(0, 2000, 50))
    Z = clf.decision_function(np.c_[yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("IsolationForest")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

    b1 = plt.scatter(timeseries.index, timeseries[:, [0]], c='white')
    # b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
    plt.show()

# dateparse = lambda dates: pd.datetime.strptime(dates, '%H:%M:%S')
df = pd.read_csv("testFitBit.csv", parse_dates=['time'], index_col='time')
df.index = pd.to_datetime(df.index, format='%H:%M:%S')
df.drop(df.columns[[0]], axis=1, inplace=True)
length = 96

timeseries = df
#
print timeseries.index
# #
# original_headers = list(df.columns.values)
# time = df.iloc[:, 1].as_matrix()
# steps = df.iloc[:, 2].as_matrix()
#
# timeseries = pd.Series(steps, index=time)
# x = timeseries.reshape(length, 1)
#
# print timeseries.interpolate()
#

# rng = np.random.RandomState(42)

clf = IsolationForest(max_samples=96, random_state=42)
clf.fit(timeseries)
#
y_pred_train = clf.predict(timeseries)
# y_pred_test = clf.predict([1000])
#
print y_pred_train
# print y_pred_test
result = timeseries
result['prediction'] = y_pred_train
# print result

xx, yy = np.meshgrid(np.linspace(0, 24, 50), np.linspace(0, 2000, 50))
Z = clf.decision_function(np.c_[yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

# b1 = plt.scatter(np.arange(96), timeseries.iloc[:, [0]], c='white')
# b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
plt.show()










# df = pd.DataFrame(y_pred_train)
# df.to_csv("isoforest.csv")
#
# plt.figure(1)
# Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
# Z1 = Z1.reshape(xx1.shape)
# legend1["SVM"] = plt.contour(xx1, yy1, Z1, levels=[0], linewidths=2, colors='m')
#
# legend1_values_list = list(legend1.values())
# legend1_keys_list = list(legend1.keys())
#
# plt.figure(1)  # two clusters
# plt.title("Outlier detection on a real data set (boston housing)")
# plt.scatter(X1[:, 0], X1[:, 1], color='black')
# bbox_args = dict(boxstyle="round", fc="0.8")
# arrow_args = dict(arrowstyle="->")

# plt.show()
