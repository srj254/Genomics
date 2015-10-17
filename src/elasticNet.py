#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
'''
###############################################################################
# generate some sparse data to play with
np.random.seed(42)

n_samples, n_features = 50, 200
X = np.random.randn(n_samples, n_features)
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[10:]] = 0  # sparsify coef
y = np.dot(X, coef)

# add noise
y += 0.01 * np.random.normal((n_samples,))

# Split data in train set and test set
n_samples = X.shape[0]
X_train, y_train = X[:n_samples / 2], y[:n_samples / 2]
X_test, y_test = X[n_samples / 2:], y[n_samples / 2:]

###############################################################################
# Lasso
from sklearn.linear_model import Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)

###############################################################################
# ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

plt.plot(enet.coef_, label='Elastic net coefficients')
plt.plot(lasso.coef_, label='Lasso coefficients')
plt.plot(coef, '--', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
#plt.show()

#############################################################################
lol = []
for i in range(50):
    l = [i]*10
    lol.append(l)

lol = np.round(np.random.randn(50, 10)*10)
print "lol:", lol.shape
print lol

W = np.zeros(10)
W[0] = 1.0
W[1] = 2.0
W[6] = 3.0
W[7] = 4.0
W = np.matrix(W)
print "W: ", W.shape
#W = W[np.newaxis, :].T
#print W
#print W.shape

e = 1.44 * np.random.randn(1, 50) + 0
print "e: ", e.shape
e = np.squeeze(np.asarray(e))

Y = np.dot(lol, W.T)
Y = np.squeeze(np.asarray(Y))
print Y
for i, val in enumerate(Y):
    Y[i] = Y[i] + e[i]
print Y

from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()
regr.fit(lol, Y)
print "Line Reg coef_:", regr.coef_

enet = ElasticNet(l1_ratio=1)
enet.fit(lol, Y)
print "Elastic Net coef_:", (enet.coef_)
#############################################################################
'''

import sys
import numpy as np

mF = sys.argv[1]
gF = sys.argv[2]

miDF = pd.DataFrame.from_csv(mF)
print "miDF:", miDF.values.shape
geDF = pd.DataFrame.from_csv(gF)
geDF1 = geDF[0:188]
print "geDF:",  geDF1[geDF1.columns[0]].values.shape

enet = ElasticNet(l1_ratio=1, max_iter=1000)
print (miDF.values).shape
print (geDF1[geDF1.columns[0]].values).shape

enet.fit(miDF.values, geDF1[geDF1.columns[0]].values)
print "coef_:", (enet.coef_).shape
hist = (geDF1[geDF1.columns[0]].values - np.dot(miDF.values, enet.coef_))

with open(sys.argv[1]+"hist.csv", 'w') as f:
    for item in hist:
        f.write(str(item))
        f.write('\n')
'''
#Code to plot if matplot lib exists

myvalues = []
with open("miRNAhistRC.csv") as f:
    for line in f.readlines():
        myvalues.append(float(line))

print myvalues
logmyValues = np.log10(myvalues)
plt.hist(myvalues)
plt.xlabel("Bins")
plt.ylabel("Frequency")
plt.savefig("Values_miRNARC.png", format='jpg');
plt.close()

plt.hist(logmyValues)
plt.xlabel("Bins")
plt.ylabel("Frequency")
plt.savefig("logValues_miRNARC.png", format='jpg');
plt.close()

myvalues = []
with open("hg19miRNAhistRC.csv") as f:
    for line in f.readlines():
        myvalues.append(float(line))

print myvalues
logmyValues = np.log10(myvalues)
plt.hist(myvalues)
plt.xlabel("Bins")
plt.ylabel("Frequency")
plt.savefig("Values_hg19miRNARC.png", format='jpg');
plt.close()

plt.hist(logmyValues)
plt.xlabel("Bins")
plt.ylabel("Frequency")
plt.savefig("logValues_hg19miRNARC.png", format='jpg');
plt.close()
'''





