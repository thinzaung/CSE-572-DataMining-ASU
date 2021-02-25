import numpy as np
import csv
import pandas as pd
from scipy.fftpack import fft, ifft
from sklearn import svm
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle

def read_data():
    df1 =[]
    with open("./test.csv",'rt')as f:
        data = csv.reader(f)
        for r in data:
            df1.append(r)
    return df1

def convert(y):
    df = []
    size_y = len(y)
    for i in range (size_y):
        y[i] = y[i][:24]
        y[i] = y[i][::-1]
        if (len(y[i])!= 24):
            df.append(i)
        elif 'NaN' in y[i]:
            df.append(i)
    for j in range (len(df),0,-1):
        del y[df[j-1]]
    return y

def diff_feature(df):
    #print("this is df lenght: ", len(df))
    average = sum(df)/len(df)
    sum_df = []
    avg_df = []
    size = 5
    for x in range(len(df)-1):
        data = df[x+1]-df[x]
        sum_df.append(data)
    np.asarray(sum_df)

    for i in range(int(len(df)/size)):
        if i != (int((len(df)/size)-1)):
            data = np.average(sum_df[(i*6):(i*6)+24])
        avg_df.append(data)
    avg_df.append(average)
    avg_df = np.asarray(avg_df)
    return avg_df

#fast fourier transform
def fast_fourier_transform(df):
    ff = 2.0/30 * np.abs(fft(df))
    #print(ff)
    ff = np.delete(ff,0)
    ff = np.unique(ff)
    max_ff = np.partition(ff,-6)[-6:]
    max_ff = np.asarray(max_ff)
    return max_ff

def test_fun():

    df= read_data()
    df= convert(df)
    feature_matrix = np.vstack((df))
    #print("thisi s Feature Matrix", feature_matrix)


    clf = pickle.load(open('./model.pickle', 'rb'))
    result = clf.predict(feature_matrix)
    result = result.transpose()
    np.savetxt('Result.csv', result, fmt="%d", delimiter=",")


if __name__ == '__main__':
   test_fun()
