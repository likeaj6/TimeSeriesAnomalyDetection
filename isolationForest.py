import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import datetime
from sklearn.ensemble import IsolationForest


#plot original timeseries data, along with rolling mean and std
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

#read fitbit data, parse into timeseries data
df = pd.read_csv("testFitBit.csv", parse_dates=['time'], index_col='time')
df.index = pd.to_datetime(df.index, format='%H:%M:%S')
#delete original index column, use time periods instead
df.drop(df.columns[[0]], axis=1, inplace=True)
#eventually change length to be extracted automatically
length = 96

#make a copy of dataframe
timeseries = df

#hyper-parameter, can adjust
rng = np.random.RandomState(42)

#initialize isolation forest with random state
clf = IsolationForest(max_samples=96, rng)

#fit classifier on data
clf.fit(timeseries)
#
#predict using training data
y_pred_train = clf.predict(timeseries)

#predict using test data
# y_pred_test = clf.predict([1000])

#training accuracy
print y_pred_train
# print y_pred_test

result = timeseries
result['prediction'] = y_pred_train

print result


###unfinished###
# still need to figure out how to plot out decision_function
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
