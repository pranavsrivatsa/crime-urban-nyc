# This modeling script was abondoned due to its poor results and the lack of importance of crime classification in the project.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load datasets
crime_data = pd.read_csv("../code/sampled_crime_data.csv")
urban_data = pd.read_csv("../code/filtered_urban_data.csv")

# Preprocess Urban Development Data
urban_data['Filing Date'] = pd.to_datetime(urban_data['Filing Date'])
urban_summary = urban_data.groupby(['BOROUGH']).size().reset_index(name='active_projects')

# Merge crime data with urban data
crime_data['CMPLNT_FR_DT'] = pd.to_datetime(crime_data['CMPLNT_FR_DT'])
crime_data = crime_data.merge(urban_summary, left_on='BORO_NM', right_on='BOROUGH', how='left')

# Feature Engineering
crime_data['DayOfWeek'] = crime_data['CMPLNT_FR_DT'].dt.dayofweek  # Add day of the week
crime_data = crime_data.drop(columns=['CMPLNT_FR_DT', 'BOROUGH'])

# Combine rare classes into "OTHER" for better class balance
threshold = 50  # Minimum number of samples for a class
crime_data['OFNS_DESC'] = crime_data['OFNS_DESC'].fillna('UNKNOWN')
value_counts = crime_data['OFNS_DESC'].value_counts()
crime_data['OFNS_DESC'] = crime_data['OFNS_DESC'].apply(
    lambda x: x if value_counts[x] > threshold else 'OTHER'
)

# Select relevant features and target
features = ['BORO_NM', 'LAW_CAT_CD', 'PREM_TYP_DESC', 'DayOfWeek', 'active_projects']
target = 'OFNS_DESC'  # Crime type
crime_data = crime_data.dropna(subset=[target])  # Drop missing target values

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(crime_data[target])

# Preprocessing pipelines
# Pipeline for MultinomialNB: No scaling
preprocessor_nb = ColumnTransformer(
    transformers=[
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore'))
        ]), ['BORO_NM', 'LAW_CAT_CD', 'PREM_TYP_DESC']),
        ('num', SimpleImputer(strategy='constant', fill_value=0), ['active_projects', 'DayOfWeek'])
    ],
    remainder='passthrough'
)

# Pipeline for other classifiers: Include scaling
preprocessor_scaled = ColumnTransformer(
    transformers=[
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore'))
        ]), ['BORO_NM', 'LAW_CAT_CD', 'PREM_TYP_DESC']),
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value=0)),
            ('scale', StandardScaler())
        ]), ['active_projects', 'DayOfWeek'])
    ],
    remainder='passthrough'
)

# Preprocess data for both pipelines
X_nb = preprocessor_nb.fit_transform(crime_data[features])  # For MultinomialNB
X_scaled = preprocessor_scaled.fit_transform(crime_data[features])  # For others

# Split data
X_nb_train, X_nb_test, y_train, y_test = train_test_split(X_nb, y, test_size=0.2, random_state=42)
X_scaled_train, X_scaled_test, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    "Naive Bayes": (MultinomialNB(), X_nb_train, X_nb_test),
    "Decision Tree": (DecisionTreeClassifier(), X_scaled_train, X_scaled_test),
    "Neural Network (MLP)": (MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                                           solver='adam', max_iter=300, random_state=42),
                             X_scaled_train, X_scaled_test)
}

# Train models and evaluate on the test set
results = {}
for name, (clf, X_train, X_test) in classifiers.items():
    print()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

# Report results
print("Classifier Performance (Post-Hoc Evaluation on Test Set):")
for clf_name, metrics in results.items():
    print(f"\n{clf_name}:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.4f}")
