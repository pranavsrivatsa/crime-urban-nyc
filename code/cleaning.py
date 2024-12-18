# The cleaning of Book1 was done by connecting kaggle to jupyter due to its size,
# hence the dataset is not avaliable in this repo, but it can be found at
# https://www.kaggle.com/code/brunacmendes/new-york-crime-analysis/notebook

import pandas as pd

file_path = "Book1.csv"
crime_df = pd.read_csv(file_path, low_memory=False)

# Step 1: Convert CMPLNT_FR_DT to datetime
crime_df['CMPLNT_FR_DT'] = pd.to_datetime(crime_df['CMPLNT_FR_DT'], errors='coerce')

# Step 2: Filter data to include only complaints from 2016 to 2019
df_filtered = crime_df[(crime_df['CMPLNT_FR_DT'] >= '2016-01-01') & (crime_df['CMPLNT_FR_DT'] <= '2019-12-31')]

# Step 3: Define essential columns
essential_columns = [
    'BORO_NM',  # Borough name
    'Latitude',  # Latitude for geographic analysis
    'Longitude',  # Longitude for geographic analysis
    'LAW_CAT_CD',  # Crime severity: felony, misdemeanor, violation
    'OFNS_DESC',  # Offense description
    'CMPLNT_FR_DT',  # Complaint date
    'PREM_TYP_DESC',  # Type of premises
    'PATROL_BORO',  # Patrol region
    'VIC_AGE_GROUP',  # Victim's age group
    'VIC_RACE',  # Victim's race
    'VIC_SEX',  # Victim's gender
]

# Step 4: Filter the DataFrame to keep only the essential columns
filtered_crime_df = df_filtered[essential_columns]

# Step 5: Randomly pick 100,000 rows
if len(filtered_crime_df) > 100000:
    sampled_crime_df = filtered_crime_df.sample(n=100000, random_state=42)
else:
    sampled_crime_df = filtered_crime_df  # If less than 100k rows, use all rows

# Display the number of rows in the sampled dataset
print(f"Number of rows in the sampled DataFrame: {len(sampled_crime_df)}")

# Step 6: Save the sampled dataset to a CSV file
sampled_crime_df.to_csv('/kaggle/working/sampled_crime_data.csv', index=False)
print("Sampled dataset saved as '/kaggle/working/sampled_crime_data.csv'")



# Cleaning of urban dataset

file_path = "../data/Book2.csv"
urban_df = pd.read_csv(file_path, low_memory=False)

urban_df['Filing Date'] = pd.to_datetime(urban_df['Filing Date'], errors='coerce')
urban_df['Issuance Date'] = pd.to_datetime(urban_df['Issuance Date'], errors='coerce')
urban_df['Expiration Date'] = pd.to_datetime(urban_df['Expiration Date'], errors='coerce')

# Filter data to include only rows from 2016 to 2019
urban_df = urban_df[
    (urban_df['Filing Date'] >= '2016-01-01') & (urban_df['Filing Date'] <= '2019-12-31')
]

# Define essential columns for urban development
essential_columns_urban = [
    'BOROUGH',              # Borough name
    'LATITUDE',             # Latitude for geographic analysis
    'LONGITUDE',            # Longitude for geographic analysis
    'Filing Date',          # Date of filing permits
    'Issuance Date',        # Date of issuance
    'Expiration Date',      # Date of permit expiration
    'Permit Status',        # Status of the permit
    'Job Type',             # Type of urban development job
    'Residential',          # Indicates residential development
    'Non-Profit',           # Indicates non-profit ownership
    'Owner\'s Business Name', # Name of owner business
    'Site Fill',            # Details on site usage
    'Street Name'
]

filtered_urban_df = urban_df[essential_columns_urban]

print("Filtered Urban Dataset (First 5 Rows):")
print(filtered_urban_df.head())

output_file_path = "../../Documents/GitHub/urban/filtered_urban_data.csv"
filtered_urban_df.to_csv(output_file_path, index=False)
print(f"Filtered urban dataset saved locally as '{output_file_path}'")
