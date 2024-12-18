# Spatio Temporal Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

# Load datasets
crime_data = pd.read_csv('../../Documents/GitHub/urban/sampled_crime_data.csv')
urban_data = pd.read_csv('../../Documents/GitHub/urban/filtered_urban_data.csv')

# Step 1: Preprocessing the Urban Dataset
# Convert 'Filing Date' to datetime
urban_data['Filing Date'] = pd.to_datetime(urban_data['Filing Date'], errors='coerce')

# Extract month and year for temporal analysis
urban_data['Year'] = urban_data['Filing Date'].dt.year
urban_data['Month'] = urban_data['Filing Date'].dt.month

# Group by borough and time (monthly and yearly)
urban_monthly = urban_data.groupby(['BOROUGH', 'Year', 'Month']).size().reset_index(name='Urban_Project_Count')
urban_yearly = urban_data.groupby(['BOROUGH', 'Year']).size().reset_index(name='Urban_Project_Count')

# Step 2: Preprocessing the Crime Dataset
# Convert 'CMPLNT_FR_DT' to datetime
crime_data['CMPLNT_FR_DT'] = pd.to_datetime(crime_data['CMPLNT_FR_DT'], errors='coerce')

# Extract month and year for temporal analysis
crime_data['Year'] = crime_data['CMPLNT_FR_DT'].dt.year
crime_data['Month'] = crime_data['CMPLNT_FR_DT'].dt.month

# Group by borough and time (monthly and yearly)
crime_monthly = crime_data.groupby(['BORO_NM', 'Year', 'Month']).size().reset_index(name='Crime_Count')
crime_yearly = crime_data.groupby(['BORO_NM', 'Year']).size().reset_index(name='Crime_Count')

# Step 3: Merge Both Datasets
# Monthly Merge
merged_monthly = pd.merge(urban_monthly, crime_monthly,
                          left_on=['BOROUGH', 'Year', 'Month'],
                          right_on=['BORO_NM', 'Year', 'Month'],
                          how='inner').drop(columns='BORO_NM')

# Yearly Merge
merged_yearly = pd.merge(urban_yearly, crime_yearly,
                         left_on=['BOROUGH', 'Year'],
                         right_on=['BORO_NM', 'Year'],
                         how='inner').drop(columns='BORO_NM')

# Step 4: Correlation Analysis
def compute_correlation(data, method='pearson'):
    """
    Computes correlation between urban project counts and crime counts.
    """
    correlation_results = {}
    for borough in data['BOROUGH'].unique():
        borough_data = data[data['BOROUGH'] == borough]
        if len(borough_data) > 1:
            if method == 'pearson':
                corr, _ = pearsonr(borough_data['Urban_Project_Count'], borough_data['Crime_Count'])
            elif method == 'spearman':
                corr, _ = spearmanr(borough_data['Urban_Project_Count'], borough_data['Crime_Count'])
            else:
                raise ValueError("Unsupported method")
            correlation_results[borough] = corr
    return correlation_results

# Compute Pearson and Spearman correlations for monthly data
monthly_pearson_corr = compute_correlation(merged_monthly, method='pearson')
monthly_spearman_corr = compute_correlation(merged_monthly, method='spearman')

# Compute Pearson and Spearman correlations for yearly data
yearly_pearson_corr = compute_correlation(merged_yearly, method='pearson')
yearly_spearman_corr = compute_correlation(merged_yearly, method='spearman')

# Step 5: Visualization

# Plot monthly trends for a sample borough (e.g., Manhattan)
borough_example = 'MANHATTAN'
manhattan_data = merged_monthly[merged_monthly['BOROUGH'] == borough_example]

plt.figure(figsize=(12, 6))
plt.plot(manhattan_data['Year'].astype(str) + '-' + manhattan_data['Month'].astype(str),
         manhattan_data['Urban_Project_Count'], label='Urban Project Count', marker='o')
plt.plot(manhattan_data['Year'].astype(str) + '-' + manhattan_data['Month'].astype(str),
         manhattan_data['Crime_Count'], label='Crime Count', marker='x')
plt.xticks(rotation=45)
plt.title(f"Urban Development vs Crime in {borough_example} (Monthly)")
plt.xlabel("Time (Year-Month)")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()
plt.show()

# Heatmap for yearly correlations across boroughs
corr_df = pd.DataFrame({
    'Borough': list(yearly_pearson_corr.keys()),
    'Pearson_Correlation': list(yearly_pearson_corr.values()),
    'Spearman_Correlation': list(yearly_spearman_corr.values())
})
corr_df = corr_df.set_index('Borough')

plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Urban Development and Crime Counts (Yearly)")
plt.show()

# Print Results
print("Monthly Pearson Correlation:", monthly_pearson_corr)
print("Monthly Spearman Correlation:", monthly_spearman_corr)
print("\nYearly Pearson Correlation:", yearly_pearson_corr)
print("Yearly Spearman Correlation:", yearly_spearman_corr)