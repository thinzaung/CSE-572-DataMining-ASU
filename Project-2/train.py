import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft
from sklearn import svm
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle

#Read cgm inputs
def read_cgm(fname):

    df_cgm_col = ['Index','Date','Time','Sensor Glucose (mg/dL)']
    df_cgm = pd.read_csv(fname,sep=',', usecols=df_cgm_col)
    df_cgm['TimeStamp'] = pd.to_datetime(df_cgm['Date'] + ' ' + df_cgm['Time'])
    df_cgm['CGM'] = df_cgm['Sensor Glucose (mg/dL)']
    df_cgm = df_cgm[['Index','TimeStamp','CGM','Date','Time']]
    df_cgm = df_cgm.replace('',np.nan)
    df_cgm = df_cgm.replace('NaN',np.nan)
    df_cgm = df_cgm.dropna()
    df_cgm.reset_index(drop=True, inplace=True)
    #df_cgm = df_cgm.bfill(axis=1)
    df_cgm = df_cgm[df_cgm['CGM'].notna()]

    return df_cgm

#Read Insulin inputs
def read_insluin(fname):
    df_ins_cols = ['Index', 'Date', 'Time', 'BWZ Carb Input (grams)']
    df_ins = pd.read_csv(fname, usecols=df_ins_cols)
    st_time = df_ins[df_ins['BWZ Carb Input (grams)'] > 0]
    st_time['TimeStamp'] = pd.to_datetime(st_time['Date'] + ' ' + st_time['Time'])
    st_time = st_time[['Index', 'TimeStamp', 'BWZ Carb Input (grams)']]

    return st_time

def cal_meal_time(df_st,df_cgm,meal):
    ### calculate meal/nomeal times ######
    df_mtime = []
    for x in df_st.index:
        df_mtime.append([df_st['TimeStamp'][x] + pd.DateOffset(hours=-0.5),
                         df_st['TimeStamp'][x] + pd.DateOffset(hours=+1.5)])

    df_nmtimes = []
    start_time = df_cgm.TimeStamp[0]
    max_time = df_cgm['TimeStamp'][df_cgm.index[-1]]
    end_time = df_cgm.TimeStamp[0]
    for x in range(len(df_mtime)):
        nomeal_end_time = df_mtime[x][0]
        df_nmtimes.append([start_time, nomeal_end_time])
        start_time = df_mtime[x][1]

    df_nmtimes.append([start_time, end_time])

    if meal == 'meal':
        #compute meal start time
        df_m = []
        for x in range(len(df_mtime)):
            data = df_cgm.loc[(df_cgm['TimeStamp'] >= df_mtime[x][0]) & (df_cgm['TimeStamp'] < df_mtime[x][1])]['CGM']
            df_m.append(data)

        df_mf = []
        for x in df_m:
            if len(x) == 24:
                df_mf.append(x)

        #print(df_meal)
        return df_mf
    else:

        df_nm = []
        for y in range(len(df_nmtimes)):
            data = df_cgm.loc[(df_cgm['TimeStamp'] > df_nmtimes[y][0]) & (df_cgm['TimeStamp'] < df_nmtimes[y][1])]['CGM']
            df_nm.append(data)
        #print(df_nm)
        df_nm2 = []
        for x in range(len(df_nm)):
            data = []
            for y in df_nm[x].index:
                data.append(df_nm[x][y])
                if len(data) == 24:
                    break
            df_nm2.append(data)

        df_nmf = []
        for y in df_nm2:
            if len(y) == 24:
                df_nmf.append(y)
        #print(df_nmf)
        #df_nmeal = []
        #for y in range(len(df_nmf)):
        #    data = []
        #    for z in df_nmf[0].index:
        #        data.append(df_nmf[0][z])
        #    df_nmeal.append(data)
        #print(df_nmeal)
        return df_nmf

def main_fun():
    df_cgm1 = read_cgm('./CGMData.csv')
    df_cgm2 = read_cgm('./CGM_patient2.csv')

    begin_time1 = read_insluin('./InsulinData.csv')
    begin_time2 = read_insluin('./Insulin_patient2.csv')

    df_meal1 = cal_meal_time(begin_time1,df_cgm1,'meal')
    df_meal2 = cal_meal_time(begin_time2,df_cgm2,'meal')
    df_meal = np.vstack((df_meal1, df_meal2))


    df_no_meal1 = cal_meal_time(begin_time1, df_cgm1, 'nomeal')
    df_no_meal2 = cal_meal_time(begin_time2,df_cgm2, 'nomeal')
    df_no_meal = np.vstack((df_no_meal1,df_no_meal2))

    df_meal_array = np.asarray(df_meal, dtype=np.int)
    df_no_meal_array = np.asarray(df_no_meal, dtype=np.int)
#    print(df_no_meal_array)

    #df_no_meal_array
    label1 = np.ones(len(df_meal_array))
    label2 = np.zeros(len(df_no_meal_array))

    feature_matrix = np.vstack((df_meal_array, df_no_meal_array))
    label = np.hstack((label1, label2))
    #print(label)
    feature_matrix_label = np.column_stack([feature_matrix, label])
    #print(feature_matrix_label)

    ###### Train Different Classification Model ######


    X = feature_matrix_label[:, 0:24]
    y = feature_matrix_label[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28)
    grid_search_params = {
        'min_samples_split': [2, 5, 10],
        'n_estimators': [200, 400, 600, 800, 1000]
    }
    clfrf = RandomForestClassifier(n_estimators=100)
    clfrf = GridSearchCV(clfrf, grid_search_params, cv=10)
    clfrf.fit(X_train, y_train)

    pickle.dump(clfrf, open('./model.pickle', 'wb'))


if __name__ == '__main__':
   main_fun()
