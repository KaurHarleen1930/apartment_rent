Apartment Rent Data, Machine_Learning_Project.py

Data Source
Originally sourced from Kaggle, data has been preprocessed and stored in GitHub to prevent system overload:

pythonCopy# Line 43



data_url = "https://github.com/KaurHarleen1930/apartment_rent-private-repo/raw/refs/heads/main/apartments_for_rent_classified_100K.csv"
# please uncomment line 45 and comment line 43 if you want to read data locally and not from github
# url = 'apartments_for_rent_classified_100K.csv'

Project Structure
Single Python file containing multiple phases of analysis:

Phase 1: EDA and Feature Engineering

1. Initial Analysis (Lines 20-150)

2. Data cleaning
3. Duplicate removal
4. Null value handling


5. Static Plot Analysis which covers (Lines 107-646)

    Univariate analysis
    Bivariate analysis
    Multivariate numerical features analysis
    Categorical feature analysis
    Statistical analysis
    Correlation and covariance analysis


6. Location Feature Addition (Lines 667-786)
pythonCopy# Commented section: Geopy API integration
# Runtime: ~2 days for 90k records due to API limitations
# Latitude and Longitude details fetch
# combine_github_csvs()


Loading and concatenating 9 CSV files containing processed location data
GitHub data retrieval
url = "https://raw.githubusercontent.com/KaurHarleen1930/apartment_rent-private-repo/refs/heads/main/ML/combined_geocoded_data.csv"
# please uncomment line 792 and comment line 790 if you want to read data locally
# url = 'combined_geocoded_data.csv'

Location Feature Analysis (Lines 791-805)

Spatial analysis
Location-based patterns (Line 807-984)


Feature Engineering (Line 807-984)

One-hot encoding
Mean price encoding for location features
Numerical feature conversion


Dimensionality Reduction & Feature Selection (Lines 1037-1188)

PCA analysis
Random forest feature importance
VIF analysis
SVD analysis



Phase 2: Regression Analysis (Lines 1195-1305)

Baseline Models

Linear Regression
Random Forest
Decision Tree
SVR
Neural Network


Advanced Regression (Line: 1310-1533)

OLS Regression
Backward elimination (p-value threshold)
Final model implementation



Phase 3: Classification Analysis (Lines 1545-2229)

pythonCopy# Several sections commented due to extensive runtime
# Optimal value calculations and SVM and Random Forest

Decision Tree Analysis

Pre-pruning
Post-pruning
Unpruned models


Multiple Classifiers

Logistic Regression
KNN (with elbow method for optimal K)
SVM (linear, polynomial, radial base kernels)
Naïve Bayes
Random Forest (including Bagging, Stacking, Boosting)
Neural Network (Multi-layered perceptron)



Phase 4: Additional Analysis (Lines 2232-2343)

K-means Clustering

Optimized k=3 using silhouette method


Association Rule Mining

Apriori algorithm implementation



Note on Execution

Several computationally intensive sections are commented out to manage runtime:

Geopy API location feature extraction
SVM kernel optimization
Random Forest optimal parameter search
One-vs-One ROC curves for ensemble methods
