import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import contingency_matrix
from math import *
import math


#Read cgm inputs
def read_cgm(fname):
    df_cgm_col = ['Index','Date','Time','Sensor Glucose (mg/dL)']
    df_cgm = pd.read_csv(fname,sep=',', usecols=df_cgm_col)
    df_cgm['TimeStamp'] = pd.to_datetime(df_cgm['Date'] + ' ' + df_cgm['Time'])
    df_cgm['CGM'] = df_cgm['Sensor Glucose (mg/dL)']
    df_cgm = df_cgm[['Index','TimeStamp','CGM','Date','Time']]
    df_cgm = df_cgm.sort_values(by=['TimeStamp'], ascending=True).fillna(method='ffill')
    df_cgm = df_cgm.drop(columns=['Date', 'Time','Index']).sort_values(by=['TimeStamp'], ascending=True)
    #print(df_cgm)

    df_cgm = df_cgm[df_cgm['CGM'].notna()]

    df_cgm.reset_index(drop=True, inplace=True)
    #print(len(df_cgm))
    return df_cgm

#Read Insulin inputs
def read_insluin(fname):
    df_ins = pd.read_csv(fname, dtype='unicode')
    df_ins['DateTime'] = pd.to_datetime(df_ins['Date'] + " " + df_ins['Time'])
    df_ins = df_ins[["Date", "Time", "DateTime", "BWZ Carb Input (grams)"]]
    df_ins['ins'] = df_ins['BWZ Carb Input (grams)'].astype(float)
    df_ins = df_ins[(df_ins.ins != 0)]
    df_ins = df_ins[df_ins['ins'].notna()]
    df_ins = df_ins.drop(columns=['Date', 'Time','BWZ Carb Input (grams)']).sort_values(by=['DateTime'], ascending=True)
    df_ins.reset_index(drop=True, inplace=True)

    df_ins_shift = df_ins.shift(-1)
    df_ins = df_ins.join(df_ins_shift.rename(columns=lambda x: x+"_lag"))
    df_ins['tot_mins_diff'] = (df_ins.DateTime_lag - df_ins.DateTime) / pd.Timedelta(minutes=1)
    df_ins['Patient'] = 'P1'

    df_ins.drop(df_ins[df_ins['tot_mins_diff'] < 120].index, inplace = True)
    df_ins = df_ins[df_ins['ins_lag'].notna()]
    #print(df_ins)

    return df_ins

    ############################################################
    ####### Calculate # of bins and get range or bins ##########
    ############################################################
def cal_bins(df_ins):
    df_bins = df_ins['ins']
    #print("insulin lenght" , len(df_bins))
    max_val = df_bins.max()
    min_val = df_bins.min()
    bins = int((max_val - min_val)/20)

    bin_label = []
    for x in range(0,bins+1):
        bin_label.append(int(min_val + x*20))


    return bin_label,bins, min_val, max_val

def cal_gt(df_ins,x1_len):
    bin_label, bins, min_val, max_val = cal_bins(df_ins)
    df_ins['min_val'] = min_val
    df_ins['bins'] = ((df_ins['ins'] - df_ins['min_val'])/20).apply(np.ceil)

    bin_truth = pd.concat([x1_len, df_ins], axis=1)
    bin_truth = bin_truth[bin_truth['len'].notna()]

    bin_truth.drop(bin_truth[bin_truth['len'] < 30].index, inplace=True)
    df_ins.reset_index(drop=True, inplace=True)
    #print(bin_truth)

    return bin_truth

def cal_meal_time(df_ins,df_cgm):
    ### calculate meal/nomeal times ######
    df_mtime = []
    for x in df_ins.index:
        df_mtime.append([df_ins['DateTime'][x] + pd.DateOffset(hours=-0.5),
                         df_ins['DateTime'][x] + pd.DateOffset(hours=+2)])

    df_m = []
    for x in range(len(df_mtime)):
        data = df_cgm.loc[(df_cgm['TimeStamp'] >= df_mtime[x][0]) & (df_cgm['TimeStamp'] < df_mtime[x][1])]['CGM']
        df_m.append(data)

    df_ml_length = []
    df_mf = []
    y = 0
    for x in df_m:
        y = len(x)
        df_ml_length.append(y)
        if len(x) == 30:
            df_mf.append(x)

    df_length = DataFrame(df_ml_length, columns=['len'])
    df_length.reset_index(drop=True, inplace=True)

    return df_mf, df_length

def get_bins(result_labels, true_label):
    #print(result_labels)
    #for x in range(len(result_labels)):
    #    print(result_labels[x])
    bin_result = {}
    bin_result[1] = []
    bin_result[2] = []
    bin_result[3] = []
    bin_result[4] = []
    bin_result[5] = []
    bin_result[6] = []
    for i in range(len(result_labels)):
        if result_labels[i] == 0:
            bin_result[1].append(i)
        elif result_labels[i] == 1:
            bin_result[2].append(i)
        elif result_labels[i] == 2:
            bin_result[3].append(i)
        elif result_labels[i] == 3:
            bin_result[4].append(i)
        elif result_labels[i] == 4:
            bin_result[5].append(i)
        elif result_labels[i] == 5:
            bin_result[6].append(i)

    bin_1 = []
    bin_2 = []
    bin_3 = []
    bin_4 = []
    bin_5 = []
    bin_6 = []

    for i in bin_result[1]:
        bin_1.append(true_label[i])
    for i in bin_result[2]:
        bin_2.append(true_label[i])
    for i in bin_result[2]:
        bin_3.append(true_label[i])
    for i in bin_result[4]:
        bin_4.append(true_label[i])
    for i in bin_result[5]:
        bin_5.append(true_label[i])
    for i in bin_result[6]:
        bin_6.append(true_label[i])
    total = len(bin_1) + len(bin_2) + len(bin_3) + len(bin_4) + len(bin_5) + len(bin_6)

    return total, bin_1, bin_2, bin_3, bin_4, bin_5, bin_6

def calculateSSE(bin):
    if len(bin) != 0:
        SSE = 0
        avg = sum(bin) / len(bin)
        for i in bin:
            SSE += (i - avg) * (i - avg)
        return SSE
    return 0

def main_fun():
    df_cgm = read_cgm('./CGMData.csv')
    df_insulin = read_insluin('./InsulinData.csv')

    x1, x1_len = cal_meal_time(df_insulin, df_cgm)
    gt_df = cal_gt(df_insulin,x1_len)

    feature_matrix = np.vstack((x1))

    df = StandardScaler().fit_transform(feature_matrix)
    number_clusters = 6
    km = KMeans(
        n_clusters=number_clusters, random_state=0).fit(np.array(df))
    ######################################################
    #### ground truth labels #############################
    ######################################################
    ground_truth_bins = gt_df["bins"]
    #print(ground_truth_bins)
    true_labels = np.asarray(ground_truth_bins).flatten()
    for i in range(len(true_labels)):
        if math.isnan(true_labels[i]):
            true_labels[i] = 1

    ######################################################
    ########### kmean labels #############################
    ######################################################
    kmeans_labels = km.labels_
    for ii in range(len(kmeans_labels)):
        kmeans_labels[ii] = kmeans_labels[ii] + 1

    ######################################################
    ############ calculate SSE for kmean #################
    ######################################################
    total, bin_1, bin_2, bin_3, bin_4, bin_5, bin_6 = get_bins(kmeans_labels,true_labels)

    kmean_SSE = (calculateSSE(bin_1) * len(bin_1) + calculateSSE(bin_2) * len(bin_2) + calculateSSE(bin_3) * len(bin_3) + calculateSSE(bin_4) * len(bin_4) + calculateSSE(bin_5) * len(bin_5) + calculateSSE(bin_6) * len(bin_6)) / (total)

    #kmean_SSE = km.inertia_ /total
    #### calculate entropy and purity #####
    #print("true labels ####", true_labels)
    km_contingency = contingency_matrix(true_labels, kmeans_labels)
    entropy, purity = [], []
    for cluster in km_contingency:
        cluster = cluster / float(cluster.sum())
        #print("cluster #####", cluster)
        e = 0
        for x in cluster :
            if x !=0 :
                e = (cluster * [log(x, 2)]).sum()
            #else:
            #    e = cluster.sum()
        p = cluster.max()
        entropy += [e]
        purity += [p]
    counts = np.array([c.sum() for c in km_contingency])
    coeffs = counts / float(counts.sum())
    kmean_entropy = (coeffs * entropy).sum()
    kmean_purity = (coeffs * purity).sum()
    #print('kmean entropy: ', kmean_entropy)
    #print('kmean purity: ', kmean_purity)

    ######################################################
    ############ Plot DB Scan ############################
    ######################################################
    feature_new = []
    for i in feature_matrix:
        feature_new.append(i[1])

    feature_new = np.array(feature_new)
    feature_new = feature_new.reshape(-1, 1)

    X = StandardScaler().fit_transform(feature_new)
    dbscan = DBSCAN(eps=0.6, min_samples=10).fit(X)
    #X = StandardScaler().fit_transform(feature_new)
    #dbscan = DBSCAN(eps=9, min_samples=5).fit(feature_new)
    dbs_labels = dbscan.labels_
    print(dbs_labels)

    ######################################################
    ############ calculate SSE for kmean #################
    ######################################################
    total, bin_1, bin_2, bin_3, bin_4, bin_5, bin_6 = get_bins(dbs_labels,true_labels)

    dbs_SSE = (calculateSSE(bin_1) * len(bin_1) + calculateSSE(bin_2) * len(bin_2) + calculateSSE(bin_3) * len(bin_3) + calculateSSE(bin_4) * len(bin_4) + calculateSSE(bin_5) * len(bin_5) + calculateSSE(bin_6) * len(bin_6)) / (total)

    #print("dbs SSE #######", dbs_SSE)

    #### calculate entropy and purity #####
    dbs_contingency = contingency_matrix(true_labels, dbs_labels)
    entropy, purity = [], []
    for cluster in dbs_contingency:
        cluster = cluster / float(cluster.sum())
        #print("cluster #####", cluster)
        e = 0
        for x in cluster :
            if x !=0 :
                e = (cluster * [log(x, 2)]).sum()
            #else:
            #    e = cluster.sum()
        p = cluster.max()
        entropy += [e]
        purity += [p]
    counts = np.array([c.sum() for c in km_contingency])
    coeffs = counts / float(counts.sum())
    dbs_entropy = (coeffs * entropy).sum()
    dbs_purity = (coeffs * purity).sum()
    #print('dbs entropy: ', dbs_entropy)
    #print('dbs purity: ', dbs_purity)

    result = []
    result.append([kmean_SSE, dbs_SSE, kmean_entropy, dbs_entropy, kmean_purity, dbs_purity])
    result = np.array(result)
    np.savetxt('./Result.csv', result, fmt="%f", delimiter=",")



if __name__ == '__main__':
   main_fun()
