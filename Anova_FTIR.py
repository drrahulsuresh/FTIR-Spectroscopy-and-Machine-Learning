import pandas as pd
import numpy as np
import os
import glob
from scipy.interpolate import interp1d
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.multivariate.manova import MANOVA

# Function to Interpolate
def read_and_label(folder_path, label):
    """Read spectra files and assign labels."""
    data_list = []
    for i, file_path in enumerate(glob.glob(os.path.join(folder_path, '*.txt'))):
        data = pd.read_csv(file_path, sep=",", header=None, names=['Wavenumber', 'Intensity'])
        data['Label'] = label
        data['Spectrum_ID'] = f"{label}_{i}"  
        data_list.append(data)
    return pd.concat(data_list, ignore_index=True)

def interpolate_spectrum(wavenumbers, intensities, common_wavenumbers):
    """Interpolate spectrum to common wavenumbers."""
    interp_func = interp1d(wavenumbers, intensities, kind='linear', fill_value="extrapolate", bounds_error=False)
    return interp_func(common_wavenumbers)

# Files
base_path = 'F:\\TUD\\DInesh\\Serum_IRC'
class_folders = {'Benign': 0, 'Low': 1, 'Intermediate': 2, 'High': 3}
common_wavenumbers = np.linspace(start=1000, stop=4000, num=3601)

# Interpolate
all_interpolated_data = []
for folder, label in class_folders.items():
    folder_path = os.path.join(base_path, folder)
    class_data = read_and_label(folder_path, label)
    for spectrum_id, group in class_data.groupby('Spectrum_ID'):
        interpolated_intensities = interpolate_spectrum(group['Wavenumber'].values, group['Intensity'].values, common_wavenumbers)
        all_interpolated_data.append(pd.DataFrame({
            'Wavenumber': common_wavenumbers,
            'Intensity': interpolated_intensities,
            'Label': label,
            'Spectrum_ID': spectrum_id
        }))

# Dataframe
final_data = pd.concat(all_interpolated_data, ignore_index=True)
final_pivoted = final_data.pivot_table(index='Spectrum_ID', columns='Wavenumber', values='Intensity').reset_index()
final_pivoted = final_pivoted.join(final_data[['Spectrum_ID', 'Label']].drop_duplicates().set_index('Spectrum_ID'), on='Spectrum_ID')
final_pivoted.to_csv('final_prepared_spectra.tsv', sep='\t', index=False)

data = pd.read_csv('final_prepared_spectra.tsv', sep='\t')

# Function to perform ANOVA
def perform_anova(data):
    results = {}
    for col in data.columns.difference(['Spectrum_ID', 'Label']):
        groups = [group[col].dropna() for name, group in data.groupby('Label')]
        stat, p_value = f_oneway(*groups)
        results[col] = {'Statistic': stat, 'P-value': p_value, 'Test Used': 'ANOVA'}
    return pd.DataFrame(results).T
data = pd.read_csv('final_prepared_spectra.tsv', sep='\t')
anova_results = perform_anova(data)
print(anova_results)
anova_results.to_csv('anova_results.tsv', sep='\t', index=True)

# Plot
plt.rcParams.update({
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,  # Unused, as no titles are set
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18
})

sns.set(style="ticks", rc={
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Load the ANOVA results
results = pd.read_csv('anova_results.tsv', sep='\t', header=None, names=['Wavenumber', 'Statistic', 'P-value', 'Test'])
results['P-value'] = pd.to_numeric(results['P-value'], errors='coerce')

# Identify significant results
significant = results[(results['P-value'] < 0.05) & results['P-value'].notna()]

# Plotting the results
plt.figure(figsize=(10, 5))
plt.scatter(results['Wavenumber'], results['P-value'], color='gray', label='Non-significant')
plt.scatter(significant['Wavenumber'], significant['P-value'], color='red', label='Significant')
plt.axhline(0.05, color='blue', linestyle='--', label='Significance Threshold (0.05)')
plt.yscale('log')
plt.xlabel('Wavenumber')
plt.ylabel('P-value (log scale)')
plt.legend()
plt.show()


# Manova 1

data = pd.read_csv('final_prepared_spectra.tsv', sep='\t')
data.rename(columns={col: 'w_' + col.replace('.', '_') for col in data.columns if 'Label' not in col and 'Spectrum_ID' not in col}, inplace=True)
dependent_variables = ' + '.join([col for col in data.columns if col.startswith('w_')])
formula = f'{dependent_variables} ~ Label'
manova = MANOVA.from_formula(formula, data=data)
result = manova.mv_test()
print(result)

# Manova reduced through PCA

data.rename(columns={col: 'w_' + col.replace('.', '_') for col in data.columns if 'Label' not in col and 'Spectrum_ID' not in col}, inplace=True)
features = data[[col for col in data.columns if col.startswith('w_')]]
labels = data['Label']
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=10)  #25 gives better results
principal_components = pca.fit_transform(features_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, 11)])
principal_df['Label'] = labels
formula = ' + '.join([f'PC{i}' for i in range(1, 11)]) + ' ~ Label'
manova = MANOVA.from_formula(formula, data=principal_df)
result = manova.mv_test()
print(result)
