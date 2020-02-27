# Implementation of the most basic ML algorithm, a perceptron that tries to
# distinguish whether cities are on the "left" or "right" side of the Atlantic
import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Function for plotting
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(-180, 180, resolution),
                           np.arange(-90, 90, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    
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

    def train(self, samples, answers):
        ranGen = np.random.RandomState(self.randState)
        self.weight_ = ranGen.normal(loc=0.0, scale=0.01, 
                                     size=samples.shape[1]+1)
        self.errors_ = []

        # Iterate over number of iterations
        for iterN in range(self.maxIter):
            errSum = 0
            # Iterate over training samples and answers
            for sample, answer in zip(samples, answers):
                # Guess based on current weights
                guess = self.predict(sample)
                # Calculate change in weights and update
                dWeight = self.eta * (answer - guess)
                self.weight_[1:] += dWeight * sample
                self.weight_[0]  += dWeight
                # Calculate number of samples that are wrong
                errSum += int(dWeight != 0.)
            # Append errors for plotting
            self.errors_.append(errSum)
            # If no errors stop iterating
            if errSum == 0:
                break
        return self

    def net_input(self, traits):
        # Calculate sum of input*weight
        return np.dot(traits, self.weight_[1:]) + self.weight_[0]

    def predict(self, traits):
        # Predict based on current weights
        return np.where(self.net_input(traits) >= 0., 1, -1)

# Import data from file
df = pd.read_csv('./citydata.csv', header=None, delimiter=';')
# Training data
answers = df.iloc[:50, 4].values.astype(np.float)
samples = df.iloc[:50, [2,1]].values.astype(np.float)
# Database of cities
tests = df.iloc[:].values

# Create perceptron instance and train on first 50 cities
ppn = Perceptron(eta=1e-2, maxIter=500)
ppn.train(samples, answers)

#Plot convergence
plt.figure()
plt.plot(range(1, len(ppn.errors_)+1), 
         ppn.errors_, marker='o')

#Plot training sample
plt.figure()
plot_decision_regions(samples, answers, classifier=ppn)
plt.show()

#Request inputs and identify location in databse
place = input("Enter the city/town:")
country = input("Enter the country:")
locInd = np.where(
    np.logical_and(
        np.array([i.strip() for i in tests[:,0]]) == place,
        np.array([i.strip() for i in tests[:,3]]) == country
        )
    )[0]

try:
    loc = tests[locInd[0]]
except:
    print("Could not find location in database")
    exit()

#Predict side of Atlantic and print result
isIn = ppn.predict(np.array(loc[1:3], dtype=np.float))
if isIn == 1:
    result = "not on"
if isIn == -1:
    result = "on"
print("{0}, {1} is {2} the left side of the Atlantic".format(loc[0], loc[3], result))
