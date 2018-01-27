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


def train(numIter, eta, inputFile):
    with open(inputFile,'r') as tsv:
        lines = [line.strip().split('\t') for line in tsv]
    label = list()
    data = list()
    for line in lines:
        length = len(line)
        label.append(line[length-1])
        temp = [line[i] for i in range(length)]
        data.append(temp)

    label = np.array(label)
    data = np.array(data)
    weight = np.zeros(len(data[0]))

    for perIter in range(numIter):
        for i, x in enumerate(data):
            if np.dot(data[i],weight)*label(i) <= 0:
                weight = weight + eta*data[i]*label[i]
        print("current weight is {}\n".format(weight))

    return weight


def test(weight, inputFile):
    with open(inputFile,'r') as tsv:
        lines = [line.strip().split('\t') for line in tsv]
    label = list()
    data = list()
    for line in lines:
        length = len(line)
        label.append(line[length-1])
        temp = [line[i] for i in range(length)]
        data.append(temp)

    eta = 1
    numIter = 30
    errors = []

    for perIter in numIter:
        total_error = 0
        for i, x in enumerate(data):
            if np.dot(data[i],weight)*label[i] <= 0:
                total_error = total_error + np.dot(data[i],weight)*label[i]
                weight = weight + eta*data[i]*label[i]
        errors.append(total_error*(-1))

    return errors
    

def main():


if __name__ == "__main__":
    main()
