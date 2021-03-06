'''

Implement two programs:

* train
* test

The train program should implement perceptron learning:

* The program should be called train
* The program should accept three input parameters: a maximum number of iterations, a learning rate, and an input tsv file


* For each intermediate iteration, the program should print the current weights (separated by tabs) to standard error
* Given an input tsv file of n columns, where the first n-1 columns are input values and the last column is the label, the program should print to standard output the weights, separated by tabs


The test program should:

* Accept weights on standard input
* Accept an input tsv file of data points with labels to be tested
* The program should print to standard output the error rate for the test set


You should also write a short report in LaTeX that describes your implementation and graphs error rate during training for each of the provided boolean functions. You should also come up with two other datasets and do the same for them. One should be linearly separable, one should not. Both of these should have appreciably more than two inputs.

Your LaTeX file should be called paper.tex and should use the ACL 2017 style and bibstyle.

'''

import numpy as np
from matplotlib import pyplot as plt
import csv
import random

def train(numIter, eta, inputFile, first, second):
    tsv = open(inputFile, "r")
    lines = tsv.readlines()
    tsv.close()
    random.shuffle(lines)
    lines = [(line.strip()).split(',') for line in lines]
    label = list()
    data = list()
    for line in lines:
        length = len(line)
        if line[length-1] == first:
            label.append(1)
        else:
            label.append(-1)

        temp = [float(line[i]) for i in range(length-1)]
        data.append(np.array(temp))

    label = np.array(label)
    data = np.array(data)
    weight = np.zeros(len(data[0]))

    for perIter in range(numIter):
        for i,x in enumerate(data):
            if data[i].dot(weight)*label[i] <= 0:
                weight += eta*data[i]*label[i]
        print("current weight is {}\n".format(weight))

    return weight


def test(weight, inputFile, first, second):
    tsv = open(inputFile, "r")
    lines = tsv.readlines()
    tsv.close()
    random.shuffle(lines)
    lines = [line.strip().split(',') for line in lines]
    label = list()
    data = list()
    for line in lines:
        length = len(line)
        if line[length-1] == first:
            label.append(1)
        else:
            label.append(-1)
        temp = [float(line[i]) for i in range(length-1)]
        data.append(np.array(temp))

    eta = 1
    numIter = 30
    errors = []

    for perIter in range(numIter):
        total_error = 0
        for i, x in enumerate(data):
            if np.dot(data[i],weight)*label[i] <= 0:
                total_error = total_error + np.dot(data[i],weight)*label[i]
                weight += eta*data[i]*label[i]
        errors.append(total_error*(-1))
        print("the current error rate is {}".format(total_error*(-1)))

    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.savefig('foo.png')


    return errors


def main():
    weight1 = train(50, 1, "setosa-versicolor.txt", "Iris-setosa", "Iris-versicolor")
    error1 = test([1,1,1,1], "setosa-versicolor.txt", "Iris-setosa", "Iris-versicolor")

#    weight2 = train(50, 1, "versicolor-virginica.txt", "Iris-versicolor", "Iris-virginica")
#    error2 = test([1,1,1,1], "versicolor-virginica.txt", "Iris-versicolor", "Iris-virginica")

#    weight3 = train(50, 1, "setosa-virginica.txt", "Iris-setosa", "Iris-virginica")
#    error3 = test([1,1,1,1],"setosa-virginica.txt", "Iris-setosa", "Iris-virginica")



if __name__ == "__main__":
    main()


#setosa-versicolor: setosa -> 1      versicolor -> -1
#versicolor-virginica   versicolor -> 1  virginica -> -1
#setosa-virginica: setosa -> 1  virginica -> -1
