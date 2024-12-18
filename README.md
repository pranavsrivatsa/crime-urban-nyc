# Urban Computing Analysis of Crime Patterns in New York City (2016-2019)

This repository contains the code and preprocessed data for the project **Urban Computing Analysis of Crime Patterns in New York City (2016-2019)**. The study explores spatio-temporal crime patterns, clustering of crime and urban development projects, and redundancy analysis to provide insights for urban safety and planning.

## Project Overview

The goal of this project is to analyze the interplay between crime patterns and urban development projects in New York City using a combination of statistical, clustering, and geospatial techniques. The repository is structured into two primary folders:

1. `code/` - Contains Python scripts for data preprocessing, visualization, spatio-temporal analysis, clustering, and redundancy analysis.
2. `data/` - Contains preprocessed datasets and links to the original data sources.

---

## Repository Structure

### Folder: `code/`

This folder includes the following Python scripts:

1. **`cleaning.py`**  
   Preprocesses the raw urban and crime datasets, handling missing values, extracting temporal features, and filtering records for analysis.

2. **`graphs.py`**  
   Generates exploratory data analysis (EDA) visualizations, such as crime distribution across boroughs, heatmaps, and bar charts.

3. **`networks.py`**  
   Constructs and analyzes the crime network using network science techniques, including clustering coefficients and redundancy metrics.

4. **`sta.py`**  
   Performs spatio-temporal correlation analysis, calculating Pearson and Spearman correlations between urban development and crime patterns over time.

5. **`clustering.py`**  
   Applies K-Means clustering on urban development and crime datasets to identify spatial clusters. Also integrates redundancy analysis for the identified clusters.

6. **`morans.py`**  
   Attempts Moran’s I spatial autocorrelation to explore spatial relationships in crime patterns.  
   **Note:** This script was abandoned due to compatibility issues with Python versions.

7. **`modeling.py`**  
   Implements machine learning models (e.g., Naive Bayes, Decision Trees, Neural Networks) for crime classification.  
   **Note:** This script was abandoned as crime classification was not a primary focus of the project.

8. **`redundancy.py`**  
   Performs standalone redundancy analysis.  
   **Note:** This script was deprecated in favor of the integrated redundancy metrics in `clustering.py`.

---

### Folder: `data/`

This folder contains the datasets used in the analysis:

1. **`Book2.csv`**  
   The original urban development dataset containing permit data from the NYC Department of Buildings.

2. **`filtered_urban_data.csv`**  
   Preprocessed urban development dataset after filtering and cleaning.

3. **`sampled_crime_data.csv`**  
   A subset of the crime dataset, sampled and cleaned for efficient analysis.

---

## Original Data Sources

1. **Crime Dataset:**  
   Available on Kaggle at [New York Crime Analysis](https://www.kaggle.com/code/brunacmendes/new-york-crime-analysis/notebook). The dataset contains detailed records of crimes in New York City.

2. **Urban Development Dataset:**  
   Provided by the NYC Department of Buildings, accessible at [DOB Permit Issuance Dataset](https://data.cityofnewyork.us/Housing-Development/DOB-Permit-Issuance/ipu4-2q9a/about_data).

---

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/pranavsrivatsa/crime-urban-nyc.git
   cd crime-urban-nyc
   ```
