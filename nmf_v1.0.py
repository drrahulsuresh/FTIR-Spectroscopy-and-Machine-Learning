from scipy import io
import pandas as pd
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF

# Set the working directory
os.chdir('/content/drive/MyDrive/Codes/BIC_v2/')
current_directory = os.getcwd()
print("Current working directory:", current_directory)

def inspect_mat_file(file_path):
    mat = sio.loadmat(file_path)
    print(f"Keys in '{file_path}': {mat.keys()}")
    for key in mat.keys():
        if isinstance(mat[key], np.ndarray):
            print(f"Shape of '{key}': {mat[key].shape}")

# Load and inspect .mat files
mat_data = io.loadmat('01.mat')
df_input = pd.DataFrame(mat_data['ab']).T

mat_data = io.loadmat('Spectra_all_prep.mat')
df = pd.DataFrame(mat_data['AB']).T
df_reversed_spectra = df.iloc[:, ::-1]

mat_data = io.loadmat('wavenumber.mat')
df_wavenumber = pd.DataFrame(mat_data['wavenumber'])
df_reversed_wavenumber = df_wavenumber.iloc[:, ::-1]

mat_data = io.loadmat('refnames_all.mat')
df_refnames = pd.DataFrame(mat_data['refnames'])

# Process input and reference spectra
input_mat = io.loadmat('111_01.mat')
input_spectra = input_mat['ab'].T
input_wavenumbers = np.linspace(1800, 900, input_spectra.shape[1])

ref_spectra_mat = io.loadmat('Spectra_all_prep.mat')
reference_spectra = ref_spectra_mat['AB'].T
wavenumber_mat = io.loadmat('wavenumber.mat')
reference_wavenumbers = wavenumber_mat['wavenumber'][0]
reference_df = pd.DataFrame(reference_spectra, columns=reference_wavenumbers)
input_df = pd.DataFrame(input_spectra, columns=input_wavenumbers)

matched_columns = {iw: reference_wavenumbers[np.argmin(np.abs(reference_wavenumbers - iw))] for iw in input_wavenumbers}
matched_reference_df = reference_df[list(matched_columns.values())]
matched_input_df = input_df.rename(columns=matched_columns)
print(f"Matched wavenumbers (input to reference): {matched_columns}")

matched_reference_df = matched_reference_df[matched_input_df.columns]
print(matched_reference_df)
input_df.to_csv('matched_input_spectra.csv', index=False)
matched_reference_df.to_csv('matched_reference_spectra.csv', index=False)

"""## **Non-negative Matrix Factorization (NMF)**"""

X = matched_reference_df.to_numpy()
Y = matched_input_df.to_numpy()

n_components = matched_reference_df.shape[0]
nmf_model = NMF(n_components=n_components, init='random', random_state=0)
W = nmf_model.fit_transform(Y)
H = nmf_model.components_
W_df = pd.DataFrame(W)
H_df = pd.DataFrame(H, columns=matched_input_df.columns)
print("Shape of W (weights):", W_df.shape)
print("Shape of H (components):", H_df.shape)
W_df.to_csv('NMF_weights.csv', index=False)
H_df.to_csv('NMF_components.csv', index=False)

