import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv('./citydata.csv', header=None, delimiter=';')
t = df.tail(1)
h = df.head(1)

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(-90, 90, resolution),
                           np.arange(-180, 180, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(-90, 90)
    plt.ylim(-180, 180)
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
         plt.scatter(x=X[y == cl, 0],
                     y=X[y == cl, 1],
                     alpha=0.8,
                     c=colors[idx],
                     marker=markers[idx],
                     label=cl,
                     edgecolor='black')

class Perceptron(object):
    
    def __init__(self, eta=0.01, maxIter=50, ranState=1):
        self.eta       = eta
        self.maxIter   = maxIter
        self.randState = ranState

    def fit(self, samples, answers):
        ranGen = np.random.RandomState(self.randState)
        self.weight_ = ranGen.normal(loc=0.0, scale=0.01, 
                                     size=samples.shape[1]+1)
        self.errors_ = []

        for iterN in range(self.maxIter):
            errSum = 0
            guesses = []
            for sample, answer in zip(samples, answers):
                guess = self.predict(sample)
                guesses.append(guess)
                dWeight = self.eta * (answer - guess)
                self.weight_[1:] += dWeight * sample
                self.weight_[0]  += dWeight
                errSum += int(dWeight != 0.)
            self.errors_.append(errSum)
            if errSum == 0:
                break
        return self

    def net_input(self, traits):
        return np.dot(traits, self.weight_[1:]) + self.weight_[0]

    def predict(self, traits):
        return np.where(self.net_input(traits) >= 0., 1, -1)

answers = df.iloc[:49, 4].values.astype(np.float)
samples = df.iloc[:49, [1,2]].values.astype(np.float)

ppn = Perceptron(eta=1e-2, maxIter=500)
ppn.fit(samples, answers)
plt.figure()
plt.plot(range(1, len(ppn.errors_)+1), 
         ppn.errors_, marker='o')

plt.figure()
plot_decision_regions(samples, answers, classifier=ppn)

plt.figure()
plt.scatter(samples[:, 1], samples[:, 0], marker='o')
plt.ylim((-90,90))
plt.xlim((-180,180))
plt.show()

tests = df.iloc[:].values
place = input("Enter the city/town:")
country = input("Enter the country:")
locInd = np.where(
    np.logical_and(
        np.array([i.strip() for i in tests[:,0]]) == place,
        np.array([i.strip() for i in tests[:,3]]) == country
        )
    )[0]
loc = tests[locInd[0]]
isIn = ppn.predict(np.array(loc[1:3], dtype=np.float))
if isIn == 1:
    result = "not in"
if isIn == -1:
    result = "in"
print("{0}, {1} is {2} the Americas".format(loc[0], loc[3], result))