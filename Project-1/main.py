import sys
import pandas as pd
import numpy as np


def mainFun(cgmfile, insulinfile):
    pd.set_option('mode.chained_assignment', None)

    cgm_df = load_cgm_data(cgmfile)
    insulin_df = load_insulin_data(insulinfile)

    amode = 'AUTO MODE ACTIVE PLGM OFF'
    amode_ins_time = insulin_df[insulin_df['Alarm'] == amode]
    amode_ins_time = amode_ins_time[amode_ins_time['TimeStamp'] == min(amode_ins_time['TimeStamp'])]
    amode_ins_time = amode_ins_time.iloc[0]['TimeStamp']

    amode_cgm_time = cgm_df[cgm_df['TimeStamp']>amode_ins_time]
    amode_cgm_time = min(amode_cgm_time['TimeStamp'])


    df_cgm_manual = cgm_df[cgm_df['TimeStamp']<amode_cgm_time]
    df_cgm_auto = cgm_df[cgm_df['TimeStamp']>=amode_cgm_time]
    dates_manual = df_cgm_manual['Date'].unique()
    dates_auto = df_cgm_auto['Date'].unique()

    final_results = np.zeros((2,18))
    final_results[0] = calc_data(df_cgm_manual,dates_manual)
    final_results[1] = calc_data(df_cgm_auto,dates_auto)
    #print(final_results)
    #final_results *=100
    #print(final_results)
    final_results = np.around(final_results, decimals=2)
    pd.DataFrame(final_results).to_csv('Results.csv', index=False, header=None)


def load_cgm_data(cgmfile):
    df_cgm_col = ['Index','Date','Time','Sensor Glucose (mg/dL)']
    df_cgm = pd.read_csv(cgmfile,sep=',', usecols=df_cgm_col)

    df_cgm['TimeStamp'] = pd.to_datetime(df_cgm['Date'] + ' ' + df_cgm['Time'])
    df_cgm['CGM'] = df_cgm['Sensor Glucose (mg/dL)']
    df_cgm = df_cgm[['Index','TimeStamp','CGM','Date','Time']]
    df_cgm = df_cgm.replace('',np.nan)
    df_cgm = df_cgm.replace('NaN',np.nan)
    df_cgm = df_cgm.dropna()

    return df_cgm

def load_insulin_data(insulinfile):

    df_ins_cols = ['Index', 'Date', 'Time', 'Alarm']
    df_ins = pd.read_csv(insulinfile, usecols=df_ins_cols)
    df_ins['TimeStamp'] = pd.to_datetime(df_ins['Date'] + ' ' + df_ins['Time'])
    df_ins = df_ins[['Index', 'TimeStamp', 'Alarm']]
    return df_ins

def cal_per_period(df,counts):
    totalrecords = 288
    if (counts == 0):
        return np.zeros(6)
    cgm1 = len(df[df['CGM'] > 180].index) / totalrecords
    cgm2 = len(df[df['CGM'] > 250].index) / totalrecords
    cgm3 = len(df[(df['CGM'] >= 70) & (df['CGM'] <= 180)].index) / totalrecords
    cgm4 = len(df[(df['CGM'] >= 70) & (df['CGM'] <= 150)].index) / totalrecords
    cgm5 = len(df[df['CGM'] < 70].index) / totalrecords
    cgm6 = len(df[df['CGM'] < 54].index) / totalrecords
    df_period = np.array([cgm1, cgm2, cgm3, cgm4, cgm5, cgm6])
    df_period *=100
    #print(df_period)
    return df_period

def calc_data(df, dates):
    ## calculate manaul data ####
    df_all_days = []
    df_final = np.zeros(18)
    for date in dates:
        df_day = {}
        timeStampMidNight = pd.Timestamp(date)
        timeStampMorning = pd.Timestamp(date + ' ' + '06:00:00')
        wholeday = df[df['Date'] == date]
        overnight = wholeday[wholeday['TimeStamp'] < timeStampMorning]
        daytime = wholeday[wholeday['TimeStamp'] >= timeStampMorning]

        df_day['wholeday'] = wholeday
        df_day['overnight'] = overnight
        df_day['daytime'] = daytime
        df_all_days.append(df_day)

    for data in df_all_days:
        rcounts = len(data['wholeday'])
        df_final[:6]+=cal_per_period(data['overnight'],rcounts)
        df_final[6:12]+=cal_per_period(data['daytime'],rcounts)
        df_final[12:18]+=cal_per_period(data['wholeday'],rcounts)

    df_final /= len(df_all_days)
    #print(df_final)
    return df_final

if __name__ == '__main__':
    mainFun(sys.argv[1], sys.argv[2])
