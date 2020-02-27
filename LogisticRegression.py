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
    Z = classifier.predict_ans(np.array([xx1.ravel(), xx2.ravel()]).T)
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

class LogRegress(object):
    def __init__(self, learnRate=0.05, maxIter=100, randState=1):
        """A logistic regression classifier

        Inputs
            learnRate : float [0, 1]
                The learning rate for the classifier, default learRate=0.05.
            maxIter   : int
                The maximum number of iterations over the training data,
                default maxIter = 100
            randState : int
                Seed for numpy random number generator, initialises weights.
        """
        self.learnRate  = learnRate
        self.maxIter   = maxIter
        self.randState = randState

    def train(self, samples, answers):
        """ Train the logistic regression object on a data sample.

        Inputs
            samples : ndarray, shape = [nSamples, nInputs]
                Array of training samples, where nSamples is the number
                of sample cases, each with nInputs input node values.
            answers : ndarray, shape = [nSamples]
                The target values or 'answers' for each sample.
        Returns
            self : object
        """
        rand = np.random.RandomState(self.randState)
        # Random distribution of small initial weights
        self.weight_ = rand.normal(loc=0.0, scale=0.01,
                                   size=samples.shape[1]+1)
        self.cost_   = []
        # Iterate to train
        for i in range(self.maxIter):
            weighted = self.weight_input(samples)
            guesses  = self.activation_func(weighted)
            # Calculate difference between answer and guess
            diffs    = answers - guesses
            # Update the weights for each input node
            self.weight_[1:] += self.learnRate * samples.T.dot(diffs)
            self.weight_[0]  += self.learnRate * diffs.sum()
            # Calculate the negative log-likelihood (cost function)
            cost = (-answers.dot(np.log(guesses)) 
                    - ((1 - answers).dot(np.log(1 - guesses)))
                    )
            self.cost_.append(cost)
        return self

    def weight_input(self, samples):
        """Compute array where element j is the dot product 
        of weights and input node values for sample j"""
        return np.dot(samples, self.weight_[1:]) + self.weight_[0]

    def activation_func(self, weighted):
        """Calculate sigmoid function of weighted input node values,
        i.e the dot product of weights and input node values"""
        return 1. / (1. + np.exp(-np.clip(weighted, -250, 250)))

    def predict_ans(self, cases, thres=0.5):
        """Return predicted class for input samples based on current weights.
        
        Inputs
            cases : ndarray, shape = [nCases, nInputs]
                Array of cases to predict class for, where nCases 
                is the number of cases to predict answers for, 
                each with 'nInputs' input node values 
            thres : the probability theshold above which to assign a case.
                Default is thres=0.5
                
        Returns
            array of indices for which the answer is the positive case"""
        w = self.weight_input(cases)
        act = self.activation_func(w)
        positive = np.where(act >= thres, 1, 0)
        return positive

# Import data from file
df = pd.read_csv('./citydata.csv', header=None, delimiter=';')
# Training data
answers = df.iloc[:100, 4].values.astype(np.float)
samples = df.iloc[:100, [2,1]].values.astype(np.float)
# Database of cities
tests = df.iloc[:].values

# Create perceptron instance and train on first 50 cities
ppn = LogRegress(learnRate=1e-5, maxIter=500)
ppn.train(samples, answers)

#Plot convergence
plt.figure()
plt.plot(range(1, len(ppn.cost_)+1), 
         ppn.cost_, marker='o')

#Plot training sample
plt.figure()
plot_decision_regions(samples, answers, classifier=ppn)
plt.show()

#Now we can predict for any cities in database
again = True
while again:
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
    isIn = ppn.predict_ans(np.array(loc[1:3], dtype=np.float))
    if isIn == 1:
        result = "not on"
    if isIn == 0:
        result = "on"
    print("{0}, {1} is {2} the left side of the Atlantic"
            .format(loc[0], loc[3], result))

    if input("Another one? [y/n]") in ["Y", "y"]:
        again = True
    else:
        again = False
