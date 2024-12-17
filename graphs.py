import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the crime dataset
crime_data_path = "sampled_crime_data.csv"
crime_df = pd.read_csv(crime_data_path)

urban_data_path = "filtered_urban_data.csv"
urban_df = pd.read_csv(urban_data_path)

# Convert 'CMPLNT_FR_DT' to datetime
crime_df['CMPLNT_FR_DT'] = pd.to_datetime(crime_df['CMPLNT_FR_DT'], errors='coerce')

# Extract year from the complaint date
crime_df['Year'] = crime_df['CMPLNT_FR_DT'].dt.year

# Graph 1: Number of crimes per borough per year
if 'BORO_NM' in crime_df.columns and 'Year' in crime_df.columns:
    crimes_per_borough_year = crime_df.groupby(['BORO_NM', 'Year']).size().reset_index(name='Crime Count')
    crimes_pivot = crimes_per_borough_year.pivot(index='Year', columns='BORO_NM', values='Crime Count')

    crimes_pivot.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Count of Crimes per Borough for Each Year')
    plt.xlabel('Year')
    plt.ylabel('Crime Count')
    plt.xticks(rotation=45)
    plt.legend(title='Borough', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Graph 2: Severity of crimes per year
if 'LAW_CAT_CD' in crime_df.columns and 'Year' in crime_df.columns:
    severity_per_year = crime_df.groupby(['Year', 'LAW_CAT_CD']).size().reset_index(name='Crime Count')
    severity_pivot = severity_per_year.pivot(index='Year', columns='LAW_CAT_CD', values='Crime Count')

    plt.figure(figsize=(10, 6))
    sns.heatmap(severity_pivot, annot=True, fmt="d", linewidths=0.5)
    plt.title('Crime Severity Heatmap Per Year')
    plt.xlabel('Severity')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.show()

# Graph 3: Top 10 types of crime locations
if 'PREM_TYP_DESC' in crime_df.columns:
    top_premises = crime_df['PREM_TYP_DESC'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_premises.values, y=top_premises.index, palette='Reds_r')
    plt.xlabel('Count')
    plt.ylabel('Premises Type')
    plt.title('Top 10 Crime Locations')
    plt.tight_layout()
    plt.show()

# Graph 4: Number of permits filed per borough per year
if 'Filing Date' in urban_df.columns and 'BOROUGH' in urban_df.columns:
    urban_df['Year'] = pd.to_datetime(urban_df['Filing Date'], errors='coerce').dt.year
    permits_by_year_borough = urban_df.groupby(['Year', 'BOROUGH']).size().unstack(fill_value=0)

    permits_by_year_borough.plot(kind='bar', stacked=False, figsize=(12, 8))
    plt.title('Permits Per Year by Borough')
    plt.xlabel('Year')
    plt.ylabel('Number of Permits')
    plt.legend(title='Borough', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Graph 5: Top 5 crimes filed per borough per year
if 'CMPLNT_FR_DT' in crime_df.columns and 'BORO_NM' in crime_df.columns and 'OFNS_DESC' in crime_df.columns:
    top_offenses = crime_df['OFNS_DESC'].value_counts().head(5).index
    crime_df['Year'] = pd.to_datetime(crime_df['CMPLNT_FR_DT'], errors='coerce').dt.year
    offenses_per_year_borough = crime_df.groupby(['Year', 'BORO_NM', 'OFNS_DESC']).size().reset_index(name='Count')
    offenses_pivot = offenses_per_year_borough.pivot_table(index=['Year', 'BORO_NM'], columns='OFNS_DESC', values='Count', fill_value=0)
    offenses_pivot_top = offenses_pivot[top_offenses]

    offenses_pivot_top.groupby('Year').sum().plot(kind='bar', figsize=(15, 8))
    plt.title('Top 5 Offense Descriptions Per Year Across Boroughs')
    plt.xlabel('Year')
    plt.ylabel('Count of Offenses')
    plt.legend(title='Offense Description', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Graph 6: Distribution of urban development vs crime
plt.figure(figsize=(12, 8))
plt.scatter(crime_df['Longitude'], crime_df['Latitude'], c='red', s=10, alpha=0.5, label='All Crime Locations')
plt.scatter(urban_df['LONGITUDE'], urban_df['LATITUDE'], c='blue', s=10, alpha=0.5, label='All Urban Development Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Comparison of Crime and Urban Development Locations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
