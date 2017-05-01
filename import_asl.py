
# coding: utf-8

# In[ ]:

import pandas as pd
import glob
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
import re
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def import_folder(folder):
    filenames = glob.glob("data/" + folder + "/*.tsd")

    files = [file.split('\\')[1].split('.')[0] for file in filenames]

    signs = pd.DataFrame()
    for filename in files:
        df = pd.read_table('data/' + folder +'/' + filename + '.tsd', header=None)
        label = filename.split('-')[0]
        ID = filename.rsplit('-', 1)[1]
        df['label'] = label
        df['ID'] = int(ID)
        df['folder'] = folder
        df.columns = ['x_L','y_L','z_L','roll_L','pitch_L','yaw_L','thumb_L','fore_L','middle_L','ring_L','little_L','x_R','y_R','z_R','roll_R','pitch_R','yaw_R','thumb_R','fore_R','middle_R','ring_R','little_R', 'label', 'ID','folder']
        signs = signs.append(df)
    return signs


def temporal_scale(signs, length, folder):
    IDs = signs['ID'].drop_duplicates().values
    labels = signs['label'].drop_duplicates().values
    signs_new = pd.DataFrame(columns = signs.columns)
    for ID in IDs:
        for sign in labels:

            df = signs[(signs['ID'] == ID) & (signs['label'] == sign)]

            df = df[['x_L', 'y_L', 'z_L', 'roll_L', 'pitch_L', 'yaw_L', 'thumb_L', 'fore_L',
                   'middle_L', 'ring_L', 'little_L', 'x_R', 'y_R', 'z_R', 'roll_R',
                   'pitch_R', 'yaw_R', 'thumb_R', 'fore_R', 'middle_R', 'ring_R',
                   'little_R']]

            df_new = pd.DataFrame(index=np.arange(length), columns = df.columns)

            for column in df:    
                column_resample = np.fft.irfft(np.fft.rfft(df[column]), length)
                df_new[column] = column_resample

            df_new['label'] = sign
            df_new['ID'] = ID
            df_new['folder'] = folder
            signs_new = signs_new.append(df_new)
            #original = signs_new[['x_L', 'y_L', 'z_L', 'x_R', 'y_R', 'z_R']]
            #scaled = (original - original.min())/(original.max() - original.min())
            #signs_new[['x_L', 'y_L', 'z_L', 'x_R', 'y_R', 'z_R']] = scaled
    return signs_new
    

def import_data(length):
    root='data'
    signs_all = pd.DataFrame()
    signs_all_ts = pd.DataFrame()
    dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
    for folder in dirlist:
        print(folder)
        signs = import_folder(folder)
        signs_ts = temporal_scale(signs, length, folder)
        signs_all = signs_all.append(signs)
        signs_all_ts = signs_all_ts.append(signs_ts)
    return signs_all_ts

def flatten(df):
    df2 = df
    df['frame'] = df.index + 1
    var = ['x_L', 'y_L', 'z_L', 'roll_L', 'pitch_L', 'yaw_L', 'thumb_L', 'fore_L',
                   'middle_L', 'ring_L', 'little_L', 'x_R', 'y_R', 'z_R', 'roll_R',
                   'pitch_R', 'yaw_R', 'thumb_R', 'fore_R', 'middle_R', 'ring_R',
                   'little_R']
    df = df[['x_L', 'y_L', 'z_L', 'roll_L', 'pitch_L', 'yaw_L', 'thumb_L', 'fore_L',
                   'middle_L', 'ring_L', 'little_L', 'x_R', 'y_R', 'z_R', 'roll_R',
                   'pitch_R', 'yaw_R', 'thumb_R', 'fore_R', 'middle_R', 'ring_R',
                   'little_R', 'label', 'frame']]
    df = df.pivot(columns = 'frame', index = 'label').reset_index()
    df.columns = [[str(e[0]) + str(e[1]) for e in df.columns.values.tolist()]]
    df['ID'] = df2['ID']
    df['folder'] = df2['folder']
    del df2
    return df

def create_flat_df(signs_all_ts, spatial_scale=False):
    IDs = signs_all_ts['ID'].drop_duplicates().values
    labels = signs_all_ts['label'].drop_duplicates().values
    dirlist = signs_all_ts['folder'].drop_duplicates().values
    if spatial_scale:
        signs_all_ts.ix[:,:-3] = scale(signs_all_ts.ix[:,:-3])
    signs_flat = pd.DataFrame()
    for folder in dirlist:
        for ID in IDs:
            for sign in labels:
                #print(ID, sign, folder)
                df = signs_all_ts[(signs_all_ts['ID'] == ID) & (signs_all_ts['label'] == sign) & (signs_all_ts['folder'] == folder)]
                signs_flat = signs_flat.append(flatten(df))
    return signs_flat
