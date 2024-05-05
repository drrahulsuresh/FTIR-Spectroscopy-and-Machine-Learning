import os
import scipy.io
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr, pearsonr, chi2_contingency

def cluster_k(p, k=2, init='k-means++', max_iter=300):
    data_to_cluster = p.reshape(-1, p.shape[2]) if len(p.shape) == 3 else p
    model = KMeans(n_clusters=k, init=init, n_init=10, max_iter=max_iter).fit(data_to_cluster)
    labels = model.labels_.reshape(p.shape[0], p.shape[1]) if len(p.shape) == 3 else model.labels_
    return model, labels

def shannons_entropy(data):
    data_flatten = data.flatten()
    data_flatten_no_nan = data_flatten[~np.isnan(data_flatten)]
    p_data = np.bincount(data_flatten_no_nan.astype(int)) / len(data_flatten_no_nan)
    entropy = -np.sum(p_data * np.log2(p_data + 1e-10))
    return entropy

mat_dir_path = 'Path'
mat_files = [f for f in os.listdir(mat_dir_path) if f.endswith('.mat')]
entropy_dict = {}

for mat_file in mat_files:
    mat_data = scipy.io.loadmat(os.path.join(mat_dir_path, mat_file))
    intensity_data = mat_data['spcImage']
    model, mask = cluster_k(intensity_data, k=2)
    background_label = np.argmin(np.sum(model.cluster_centers_, axis=1))
    signal_label = 1 - background_label
    signal_mask = (mask == signal_label)
    intensity_data_no_background = np.where(signal_mask[..., np.newaxis], intensity_data, np.nan)
    entropy = shannons_entropy(intensity_data_no_background)
    patient_number = mat_file.split('_')[0]
    entropy_dict[patient_number] = entropy

entropy_df = pd.DataFrame.from_dict(entropy_dict, orient='index', columns=['Shannon_Entropy'])
entropy_df.index.name = 'PatientNumber'
entropy_df.to_csv('New_entropy_values.csv')

entropy_path = "New_entropy_values.csv"
screening_path = 'file.csv'

df_entropy = pd.read_csv(entropy_path)
df_screening = pd.read_csv(screening_path)
merged_df = pd.merge(df_entropy, df_screening, on='PatientNumber', how='inner')

sns.histplot(merged_df['Shannon_Entropy'], kde=True)
plt.title('Distribution of Entropy Values')
plt.show()

correlation_matrix = merged_df[['Shannon_Entropy', 'Recurrenceyesno', 'nhg']].corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Statistical Analysis

mw_stat, mw_p = mannwhitneyu(merged_df['Shannon_Entropy'][merged_df['Recurrenceyesno'] == 1],
                             merged_df['Shannon_Entropy'][merged_df['Recurrenceyesno'] == 0])
print(f'Mann-Whitney U Test: Stat={mw_stat}, P-value={mw_p}')

# Spearman's Rank Correlation
spearman_corr, spearman_p = spearmanr(merged_df['Shannon_Entropy'], merged_df['Recurrenceyesno'])
print(f'Spearman correlation: {spearman_corr}, P-value: {spearman_p}')

# Pearson's Correlation
pearson_corr, pearson_p = pearsonr(merged_df['Shannon_Entropy'], merged_df['Recurrenceyesno'])
print(f'Pearson correlation: {pearson_corr}, P-value: {pearson_p}')

# Chi-squared Test for Independence
contingency = pd.crosstab(merged_df['Recurrenceyesno'], merged_df['nhg'])
chi2, chi_p, chi_dof, _ = chi2_contingency(contingency)
print(f'Chi-squared Test: Chi2={chi2}, P-value={chi_p}, Degrees of Freedom={chi_dof}')

