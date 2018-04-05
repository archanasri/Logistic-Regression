from __future__ import division
from collections import defaultdict
import math
import sys
import random

param = defaultdict(list)

train_file = tuple(open(sys.argv[1],"r"))
test_file = tuple(open(sys.argv[2], "r"))

f_name = [sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]]
with open('train1.txt', 'w') as out_file:
    for f in f_name:
        with open(f) as in_file:
            for line in in_file:
                out_file.write(line)

f_name = [sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[7]]
with open('train2.txt', 'w') as out_file:
    for f in f_name:
        with open(f) as in_file:
            for line in in_file:
                out_file.write(line)

f_name = [sys.argv[3], sys.argv[4], sys.argv[6], sys.argv[7]]
with open('train3.txt', 'w') as out_file:
    for f in f_name:
        with open(f) as in_file:
            for line in in_file:
                out_file.write(line)

f_name = [sys.argv[3], sys.argv[5], sys.argv[6], sys.argv[7]]
with open('train4.txt', 'w') as out_file:
    for f in f_name:
        with open(f) as in_file:
            for line in in_file:
                out_file.write(line)

f_name = [sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]]
with open('train5.txt', 'w') as out_file:
    for f in f_name:
        with open(f) as in_file:
            for line in in_file:
                out_file.write(line)

t_file1 = tuple(open("train1.txt","r"))
t_file2 = tuple(open("train2.txt","r"))
t_file3 = tuple(open("train3.txt","r"))
t_file4 = tuple(open("train4.txt","r"))
t_file5 = tuple(open("train5.txt","r"))

split_file1 = tuple(open(sys.argv[3], "r"))
split_file2 = tuple(open(sys.argv[4], "r"))
split_file3 = tuple(open(sys.argv[5], "r"))
split_file4 = tuple(open(sys.argv[6], "r"))
split_file5 = tuple(open(sys.argv[7], "r"))

def regression(t_file, weight, bias, learning_rate, trade):
    line_array = random.sample(t_file, len(t_file))
    for t in range(len(t_file)):
        line = line_array[t]
        feature = [0]*68000
        line = line.strip("\n").split()
        output = line[0]
        output = int(output)
        for i in range(len(line)):
            if i != 0:
                wording = line[i].split(":")
                feature[int(wording[0])-1] = 1
        for i in range(68000):
            try:
                a = math.exp(-1 * output * weight[i] * feature[i])
                b = (2 * weight[i])/trade
                c = (-1 * feature[i] * output * a)/(1 + a)
                d = c + b
                weight[i] = weight[i] - (learning_rate * d)
            except OverflowError:
                pass
        try:
            a = math.exp(-1 * output * bias)
            b = (2 * bias)/trade
            c = (-1 * output * a)/(1 + a)
            d = c + b
            bias = bias - (learning_rate * d)
        except OverflowError:
            pass
    return weight, bias

def dev(files, weight, bias):
    accuracy = 0.0
    instances = 0.0
    for s in range(len(files)):
        lineD = files[s]
        featureD = []
        instances = instances + 1
        lineD = lineD.strip("\n").split()
        outputD = lineD[0]
        outputD = int(outputD)
        for iD in range(len(lineD)):
            if iD != 0:
                wordingD = lineD[iD].split(":")
                m = int(wordingD[0])
                featureD.append(m-1)
        SumDev = 0
        for i in range(len(featureD)):
            SumDev = SumDev + (weight[i] * featureD[i])
        res_sum = SumDev + bias
        if res_sum < 0:
            r = -1
        else:
            r = 1
        if r == outputD:
            accuracy = accuracy + 1
    return (accuracy/instances) * 100

def main():
    weight = []
    bias = random.uniform(-0.01, 0.01)
    l_r = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    t_r = [0.1, 1, 10, 100, 1000, 10000]
    for i in range(68000):
        z = random.uniform(-0.01, 0.01)
        weight.append(z)
    for l in l_r:
        for r in t_r:
            temp_accuracy = 0.0
            for e in range(10):
                weight, bias = regression(t_file1, weight, bias, l, r)
                a = dev(split_file5, weight, bias)
                if a > temp_accuracy:
                    temp_accuracy = a
                weight, bias = regression(t_file2, weight, bias, l, r)
                a = dev(split_file4, weight, bias)
                if a > temp_accuracy:
                    temp_accuracy = a
                weight, bias = regression(t_file3, weight, bias, l, r)
                a = dev(split_file3, weight, bias)
                if a > temp_accuracy:
                    temp_accuracy = a
                weight, bias = regression(t_file4, weight, bias, l, r)
                a = dev(split_file2, weight, bias)
                if a > temp_accuracy:
                    temp_accuracy = a
                weight, bias = regression(t_file5, weight, bias, l, r)
                a = dev(split_file1, weight, bias)
                if a > temp_accuracy:
                    temp_accuracy = a
                if e == 9:
                    r = str(l)
                    m = str(c)
                    key = r + ":" + m
                    param[key] = temp_accuracy
    best = max(param, key=param.get)
    b = best.split(":")
    print "Best hyperparamerter: Learning Rate is ", b[0], "Tradeoff is ", b[1]
    print "Cross Validation Accuracy: ", param[best]
    b[0] = float(b[0])
    b[1] = float(b[1])
    weight, bias = regression(train_file, weight, bias, b[0], b[1])
    a = dev(test_file, weight, bias)
    print "Traning Accuracy: ", a
    a = dev(test_file, weight, bias)
    print "Test Accuracy: ", a

if __name__ == '__main__':
    main()
