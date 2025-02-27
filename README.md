# apartment_rent
Regression, Classification and Association rule mining performed on Apartment Rent data from Kaggle
Apartment Rent Data, Information_Vis_Static.py, Dynamic_dash.py, style.css and img.png

Data Source
Originally sourced from Kaggle, data has been preprocessed and stored in GitHub to prevent system overload:

# For Information_Vis_Static.py
pythonCopy# Line 23 data loading

url = 'https://github.com/KaurHarleen1930/apartment_rent-private-repo/raw/refs/heads/main/apartments_for_rent_classified_100K.csv'
# please uncomment line 42 and comment line 40 if you want to read data locally
# url = 'apartments_for_rent_classified_100K.csv'

Project Structure
Single Python file containing multiple phases of analysis:

1. Initial Analysis and Data Loading (Lines 40-104)
Includes - 
    Data cleaning
    Duplicate removal
    Null value handling


2. Static Plot Analysis which covers (Lines 106-529)

    Univariate analysis
    Bivariate analysis
    Multivariate numerical features analysis
    Categorical feature analysis

3. Outlier Detection and Removal (530-605)

4. Normality Test (606-679)

5. Statistical analysis and Correlation analysis (line 680 - 869)

6. Feature Encodings (Line 874-950)

    One-hot encoding
    Mean price encoding for location features
    Numerical feature conversion


7. Dimensionality Reduction and Data standarization - PCA(Lines 952- 1063)


# For Dynamic_dash.py

pythonCopy# Line 20 data loading


url = 'https://github.com/KaurHarleen1930/apartment_rent-private-repo/raw/refs/heads/main/apartments_for_rent_classified_100K.csv'
# please uncomment line 26 and comment line 24 if you want to read data locally
# url = 'apartments_for_rent_classified_100K.csv'

Create assests folder and add style.css and img.png file in it.

Project Structure
Single Python file containing multiple phases of analysis:

1. Created Sidebar Navigation Buttons

2. App Layout

3. Added different page layouts (Till line 975)
    Landing Page
    data_cleaning_layout
    outlier detection layout
    data_transformation_layout
    normality_test_layout
    dimensionality_reduction_layout
    numerical_viz_layout
    categorical_viz_layout
    statistical_analysis_layout

4. Rendered all page contents, app callback (line 976 - 1005)

5. Then each page callback (1006 - 3109)

# Note

For dynamic plot creation I have commented pairplot and scatter plots with reg line and dist plot as they were working
fine in local but were not working when deployed.





