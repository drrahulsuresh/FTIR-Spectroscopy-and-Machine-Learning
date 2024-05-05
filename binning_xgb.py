import os
import numpy as np
import scipy.io as sio
import pandas as pd
import csv
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import itertools
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import GroupKFold
import time


tic = time.perf_counter()

new_directory_path = '/content/drive/MyDrive/Codes/fasa2201/'
os.makedirs(new_directory_path)
os.chdir(new_directory_path)

final_num = 100

def binningSpectra(full_set, bin_mode, total):
    per_group = total // final_num + 1
    mean_spectra = []
    if bin_mode == 0:
        grouped_spectra = [full_set[n:n + per_group] for n in range(0, len(full_set), per_group)]
        for i in range(len(grouped_spectra)):
            mean_spectra.append(np.mean(grouped_spectra[i], axis=0))
        return np.array(mean_spectra)
    elif bin_mode == 1:
        idx = np.random.choice(range(total), total, replace=False)
        full_set_re = full_set[idx, :]
        grouped_spectra = [full_set_re[n:n + per_group] for n in range(0, len(full_set_re), per_group)]
        for i in range(len(grouped_spectra)):
            mean_spectra.append(np.mean(grouped_spectra[i], axis=0))
        return np.array(mean_spectra)

dataPath = 'path_to_data'
input_file = pd.read_csv('path_to_data')
files_dir = 'path_to_data'
files = os.listdir(files_dir)[:10]
print("List of chosen files:")
for file in files:
    print(file)

REF = pd.DataFrame()
for i, fname in enumerate(files):
    pid = int(fname[0:3])
    now = input_file['PatientNumber'] == pid 
    dat = input_file[now] 
    dat['FileName'] = fname 
    sampleno = int(fname[4:6])
    dat['SampleNo'] = sampleno
    REF = pd.concat([REF, dat])

REF.to_csv('dict_files.csv') 
fileName = REF['FileName'].tolist()
patientNo = REF["PatientNumber"].tolist()
sampleNo = REF['SampleNo'].tolist()
label = REF['nhg'].tolist()
vectPatNo, index = np.unique(patientNo, return_index=True)
label_vectPatNo = np.array(label)[index]

mat_file_path = 'path_to_file'
mat_contents = sio.loadmat(mat_file_path)
keys = mat_contents.keys()
print(keys)

for key in keys:
    print(f"Contents of '{key}':")
    print(mat_contents[key])
    print("\n")

mat = sio.loadmat(dataPath + fileName[0])
mat = mat['wavenumbers'].T
OPMat = pd.DataFrame()
Pat_vec = []
NHG = []

for i, n in enumerate(patientNo):
    try:
        mat = sio.loadmat(dataPath + fileName[i])
        spectraTissue = mat['wavenumbers'].T
        spectraTissue = spectraTissue[~np.all(spectraTissue == 0, axis=1)]
        num_spc = spectraTissue.shape[0]
        spectraTissue = binningSpectra(spectraTissue, 1, num_spc)
        if isinstance(spectraTissue, np.ndarray):
            spectraTissue = pd.DataFrame(spectraTissue)
        OPMat = pd.concat([OPMat, spectraTissue], ignore_index=True)
        num_entries = spectraTissue.shape[0]
        Pat_vec.extend([patientNo[i]] * num_entries)
        NHG.extend([label[i]] * num_entries)
    except Exception as e:
        print(f"Error processing file {fileName[i]}: {e}")

Pat_vec = [item for sublist in Pat_vec for item in sublist]
NHG = [item for sublist in NHG for item in sublist]
OPMat_array = np.array(OPMat)
bindict = {"spectraTissue": OPMat_array, "patientNo": Pat_vec, "label": NHG}
os.makedirs('output', exist_ok=True)
file_to_save = os.path.join('output', 'R_Binned_spectra.mat')
sio.savemat(file_to_save, bindict)
mat_file_path = os.path.join('output', "R_Binned_spectra.mat")
mat_contents = sio.loadmat(mat_file_path)

for key, value in mat_contents.items():
    if isinstance(value, np.ndarray):
        print(f"Key: {key}, Shape: {value.shape}")
    else:
        print(f"Key: {key}, Value: {value}")

spectraTissue = mat_contents['spectraTissue']
patientNo = mat_contents['patientNo']
y = mat_contents['label']
print("\nShape of spectraTissue:", spectraTissue.shape)
print("Sample data from spectraTissue:", spectraTissue[:5])

print("\nShape of patientNo:", patientNo.shape)
print("Sample data from patientNo:", patientNo[:5])

print("\nShape of y:", y.shape)
print("Sample data from y:", y[:5])

toc = time.perf_counter()
print("\nExecution time: ", toc - tic)



mat_file = sio.loadmat(os.path.join('output', "R_Binned_spectra.mat"))
spectraTissue = mat_file['spectraTissue']
patientNo = mat_file['patientNo'].ravel()
y = mat_file['label'].ravel()

print("Length of spectraTissue:", len(spectraTissue))
print("Length of patientNo:", len(patientNo))
print("Length of y:", len(y))
if len(y) > len(spectraTissue):
    y = y[:len(spectraTissue)]

print("Length of y after trimming:", len(y))
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Length of encoded y:", len(y_encoded))
if len(spectraTissue) == len(y_encoded):
    print("Lengths of spectraTissue and y are aligned.")
else:
    print("Lengths of spectraTissue and y are not aligned.")
    
mat_file = sio.loadmat(os.path.join('output', "R_Binned_spectra_100_per_pat_8_layers.mat"))
spectraTissue = mat_file['spectraTissue']
patientNo = mat_file['patientNo'].ravel()  # Flatten to 1D
y = mat_file['label'].ravel()  # Flatten to 1D
print("Sample entries (patientNo, y, first feature of spectraTissue):")
for i in range(min(10, len(patientNo), len(y))):  # Adjust range as needed
    print(patientNo[i], y[i], spectraTissue[i, 0])
    
print("\nLength of spectraTissue:", len(spectraTissue))
print("Length of y:", len(y))
if len(y) > len(spectraTissue):
    print("\nTrimming y to match the length of spectraTissue.")
    y_trimmed = y[:len(spectraTissue)]
    print("Length of y after trimming:", len(y_trimmed))
else:
    print("\nNo trimming required.")

if spectraTissue.shape[0] >= 5:
    spectraTissue = savgol_filter(spectraTissue, 5, 1, 0, axis=0)
else:
    print("Error: Data size along axis 0 is less than 5. Adjust the window length or handle this case.")

NHG = mat_file['label'].T
NHG = [int(x) for x in NHG]
le = LabelEncoder()
y = le.fit_transform(NHG)
patientNo = mat_file['patientNo'].T
patient_groups = patientNo.ravel()  
nsplits = 5
gkf = GroupKFold(n_splits=nsplits)
trainperf = []
testperf = []
hyp = []

for train_index, test_index in gkf.split(spectraTissue, y, groups=patient_groups):
    X_train, X_test = spectraTissue[train_index], spectraTissue[test_index]
    y_train, y_test = y[train_index], y[test_index]
    xgb_mod = xgb.XGBClassifier()
    paramspace = {
        'n_estimators': [5, 100],
        'max_depth': [5, 20],
        'eta': [0.01, 0.1, 1],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.1, 0.3, 0.8]
    }
    grid = GridSearchCV(xgb_mod, param_grid=paramspace, cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=0), verbose=2)
    mod = grid.fit(X_train, y_train)
    hyp.append(mod.best_params_)
    trainperf.append(mod.best_score_)
    preds = mod.predict(X_test)
    tp = f1_score(y_test, preds, average='macro')
    testperf.append(tp)
OP = pd.DataFrame({'train': trainperf, 'test': testperf, 'params': hyp})
OP.to_csv("output/R_XGB_performances_1000_spec_per_patient.csv", index=False)
toc = time.perf_counter()
print("Execution time: ", toc - tic)


mat_file = sio.loadmat(os.path.join('output', "R_Binned_spectra.mat"))
spectraTissue = mat_file['spectraTissue']
patientNo = mat_file['patientNo'].ravel()
y = mat_file['label'].ravel()

print("Length of spectraTissue:", len(spectraTissue))
print("Length of patientNo:", len(patientNo))
print("Length of y:", len(y))
if len(y) > len(spectraTissue):
    y = y[:len(spectraTissue)]

print("Length of y after trimming:", len(y))
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Length of encoded y:", len(y_encoded))
if len(spectraTissue) == len(y_encoded):
    print("Lengths of spectraTissue and y are aligned.")
else:
    print("Lengths of spectraTissue and y are not aligned.")

mat_file = sio.loadmat(os.path.join('output', "R_Binned_spectra_100_per_pat_8_layers.mat"))
spectraTissue = mat_file['spectraTissue']
patientNo = mat_file['patientNo'].ravel()
y = mat_file['label'].ravel()

print("Sample entries (patientNo, y, first feature of spectraTissue):")
for i in range(min(10, len(patientNo), len(y))):
    print(patientNo[i], y[i], spectraTissue[i, 0])

print("\nLength of spectraTissue:", len(spectraTissue))
print("Length of y:", len(y))

if len(y) > len(spectraTissue):
    print("\nTrimming y to match the length of spectraTissue.")
    y = y[:len(spectraTissue)]
    print("Length of y after trimming:", len(y))
else:
    print("\nNo trimming required.")

if spectraTissue.shape[0] >= 5:
    spectraTissue = savgol_filter(spectraTissue, 5, 1, 0, axis=0)
else:
    print("Error: Data size along axis 0 is less than 5. Adjust the window length or handle this case.")

NHG = mat_file['label'].T
NHG = [int(x) for x in NHG]
le = LabelEncoder()
y = le.fit_transform(NHG)

patientNo = mat_file['patientNo'].T.ravel()
gkf = GroupKFold(n_splits=5)

trainperf = []
testperf = []
hyp = []

for train_index, test_index in gkf.split(spectraTissue, y, groups=patientNo):
    X_train, X_test = spectraTissue[train_index], spectraTissue[test_index]
    y_train, y_test = y[train_index], y[test_index]

    xgb_mod = xgb.XGBClassifier()
    paramspace = {
        'n_estimators': [5, 100],
        'max_depth': [5, 20],
        'eta': [0.01, 0.1, 1],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.1, 0.3, 0.8]
    }

    grid = GridSearchCV(xgb_mod, param_grid=paramspace, cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=0), verbose=2)
    mod = grid.fit(X_train, y_train)

    hyp.append(mod.best_params_)
    trainperf.append(mod.best_score_)
    preds = mod.predict(X_test)
    tp = f1_score(y_test, preds, average='macro')
    testperf.append(tp)

OP = pd.DataFrame({'train': trainperf, 'test': testperf, 'params': hyp})
OP.to_csv("output/R_XGB_performances_1000_spec_per_patient.csv", index=False)

toc = time.perf_counter()
print("Execution time: ", toc - tic)

