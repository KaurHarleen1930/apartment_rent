#%%
from statistics import covariance
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.linalg import svd
from scipy.stats import spearmanr
from seaborn import color_palette
from sklearn import metrics
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingClassifier, \
    GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, \
    precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
pd.options.display.float_format = "{:,.2f}".format
warnings.filterwarnings("ignore")


######################################################################################################################
#PHASE 1
######################################################################################################################

#%%
url = 'https://github.com/KaurHarleen1930/apartment_rent-private-repo/raw/refs/heads/main/apartments_for_rent_classified_100K.csv'
#######please uncomment line 45 and comment line 43 if you want to read data locally
#url = 'apartments_for_rent_classified_100K.csv'
df = pd.read_csv(url, sep=";",  encoding='cp1252')
pd.set_option('display.max_columns', None)
print(df.head())

print(f"Unique values in dataset: {df.nunique()}")
#since currency has only one value
print(f"Checking for value in Currency feature: {df['currency'].value_counts()}")
#checking the categories with less unique value and checking the significance.
print(f"Checking for value in category feature: {df['category'].value_counts()}")
print(f"Checking for value in category feature: {df['fee'].value_counts()}")
print(f"Checking for value in category feature: {df['has_photo'].value_counts()}")
print(f"Checking for value in category feature: {df['pets_allowed'].value_counts()}")
print(f"Checking for value in category feature: {df['price_type'].value_counts()}")
print(f"Null values analysis:\n {df.isnull().sum()}")
#other column analysis
print(f"Title column:\n{df['title'].value_counts()}")
print(f"Body column:\n{df['body'].value_counts()}")
print(f"Price display column:\n{df[['price', 'price_display']].head()}")
print(f"Column datatype for price and price_display: {df['price'].dtype} and {df['price_display'].dtype}")
####added print for only those columns which I removed going forward..
#%%
def get_apartment_rent_data(df):
    df = df[[
        'id', 'amenities', 'bathrooms', 'bedrooms', 'fee', 'has_photo',
        'pets_allowed', 'price', 'square_feet', 'cityname', 'state',
        'latitude', 'longitude', 'source', 'time'
    ]] #removed features #'category', 'title', 'body','currency','price_display', 'price_type', 'address'

    df = df.astype({
        'amenities': 'string',
        'bathrooms': 'Float32',
        'bedrooms': 'Float32',
        'fee': 'string',
        'has_photo': 'string',
        'pets_allowed': 'string',
        'price': 'Float64',
        'square_feet': 'Int64',
        'latitude': 'Float64',
        'longitude': 'Float64',
        'source': 'string',
        'time': 'Int64'
    })
    apts = pd.DataFrame(df)
    apts['amenities'] = apts["amenities"].fillna("no amenities available")
    apts["pets_allowed"] = apts["pets_allowed"].fillna("None")
    apts['time'] = pd.to_datetime(apts['time'], errors='coerce')
    apts = apts[apts['bathrooms'].notna()]
    apts = apts[apts['bedrooms'].notna()]
    apts = apts[apts['price'].notna()]
    apts = apts[apts['latitude'].notna()]
    apts = apts[apts['longitude'].notna()]
    apts = apts[apts['cityname'].notna()]
    apts = apts[apts['state'].notna()]
    apts = apts[apts.state != 0]

    apts['state'] = apts['state'].astype("string")
    apts['cityname'] = apts['cityname'].astype("string")
    apts['time'] = pd.to_datetime(apts['time'], errors='coerce')
    print(apts.info())
    print(apts.describe())
    print(apts.isnull().sum())
    return apts

#%% univariate analysis
numerical_cols = ['bathrooms', 'bedrooms', 'price', 'square_feet']
apts = get_apartment_rent_data(df)

for col in numerical_cols:
    #histogram plot
    plt.figure(figsize=(8, 4))
    sns.histplot(apts[col], kde=True, bins=30, color='blue')
    plt.title(f"Initial Histogram plot of {col}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

    #box plot
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=apts[col], color='orange')
    plt.title(f"Boxplot of {col} to check for outliers")
    plt.xlabel(col)
    plt.show()

    #violin plot
    plt.figure(figsize=(8, 4))
    violin = sns.violinplot(
        data=apts,
        y=col,
        width=0.8,
        inner='box',
        linewidth=2,
        color='skyblue'
    )
    plt.title(f'Violin plot: Distribution of {col}', pad=20)
    plt.ylabel(col)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
#%% more univariate plots
#KDE plot
numerical_cols = ['bathrooms', 'bedrooms', 'price', 'square_feet']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.kdeplot(
        data=apts,
        x=feature,
        fill=True,
        alpha=0.6,
        linewidth=2.5,
        color=sns.color_palette("husl", 8)[i-1]
    )
    plt.title(f'KDE Plot of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

###dist plot with rug
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(
        data=apts,x
        x=feature,
        stat='density',
        kde=True,
        line_kws={'linewidth': 2},
        color=sns.color_palette("bright", 8)[i-1],
        alpha=0.3
    )
    sns.rugplot(
        data=apts,
        x=feature,
        color=sns.color_palette("bright", 8)[i-1],
        alpha=0.6
    )
    plt.title(f'Dist Plot distribution {feature} with Rug Plot')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%% Area plots
def area_plots(numerical_cols, apts):
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numerical_cols, 1):
        plt.subplot(2, 2, i)
        # Sort values for smooth area plot
        sorted_data = apts[feature].sort_values().reset_index(drop=True)

        plt.fill_between(
            range(len(sorted_data)),
            sorted_data,
            alpha=0.6,
            color=sns.color_palette("husl", 8)[i - 1],
            linewidth=2
        )
        plt.title(f'Area Plot of {feature.capitalize()}')
        plt.xlabel('Index (Sorted)')
        plt.ylabel(feature.capitalize())
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


    pivot_data = pd.pivot_table(
        apts,
        values='price',
        index='bedrooms',
        columns='bathrooms',
        aggfunc='mean'
    ).fillna(0)

    pivot_data.plot(
        kind='area',
        stacked=True,
        alpha=0.6,
        colormap='viridis'
    )
    plt.title('Stacked Area Plot: Price by Bedrooms and Bathrooms')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Price')
    plt.legend(title='Number of Bathrooms', bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    for feature in numerical_cols:
        sorted_data = apts[feature].sort_values()
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        plt.fill_between(
            sorted_data,
            cumulative,
            alpha=0.3,
            label=feature
        )

    plt.title('Cumulative Distribution Area Plot')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Proportion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

area_plots(numerical_cols, apts)

#%% bivariate numerical feature analysis

plt.figure(figsize=(12, 6))

for i, col1 in enumerate(['price', 'square_feet']):
    for j, col2 in enumerate(['bedrooms', 'bathrooms']):

        plt.scatter(apts[col2], apts[col1], alpha=0.5)
        plt.xlabel(col2)
        plt.ylabel(col1)
        plt.title(f'Scatter Plot: {col2} vs {col1}')
        plt.show()
#stacked bar plot

apts_grouped = apts.groupby('bedrooms')['price'].mean().reset_index()
plt.bar(apts_grouped['bedrooms'], apts_grouped['price'])
plt.xlabel('Number of Bedrooms')
plt.ylabel('Average Price')
plt.title('Average Price by Number of Bedrooms')
plt.show()

# Count Plot

apts['price_category'] = pd.qcut(apts['price'], 3, labels=['Low', 'Medium', 'High'])
sns.countplot(data=apts, x='bedrooms', hue='price_category')
plt.title('Count of Properties by Bedrooms and Price Category')
plt.tight_layout()
plt.show()
apts.drop(columns = 'price_category', axis=1, inplace=True)#it was created for analysis, therefore removed this attribute as of now
#%% 3d plot
fig = plt.figure(figsize=(15, 10))
plot_num = 1

for i in range(len(numerical_cols)):
    for j in range(i+1, len(numerical_cols)):
        for k in range(j+1, len(numerical_cols)):
            ax = fig.add_subplot(2, 2, plot_num, projection='3d')
            scatter = ax.scatter(apts[numerical_cols[i]],
                               apts[numerical_cols[j]],
                               apts[numerical_cols[k]],
                               cmap='viridis')
            ax.set_xlabel(numerical_cols[i])
            ax.set_ylabel(numerical_cols[j])
            ax.set_zlabel(numerical_cols[k])
            plt.colorbar(scatter)
            ax.set_title(f'3D: {numerical_cols[i]} vs {numerical_cols[j]} vs {numerical_cols[k]}')
            plot_num += 1

plt.tight_layout()
plt.show()



#%%
plt.figure(figsize=(10, 10))
correlation = apts[numerical_cols].corr()
sns.clustermap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Clustered Correlation Matrix')
plt.tight_layout()
plt.show()

#%% categorical feature analysis
from collections import Counter
categorical_cols = ['has_photo', 'pets_allowed', 'state', 'fee']

for col in categorical_cols:
    #count plot
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=apts, palette='bright', order=apts[col].value_counts().index)
    plt.title(f"Count of {col}")
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

#pie chart

colors = sns.color_palette('bright', n_colors=10)


fig = plt.figure(figsize=(20, 15))

#top 10 states
plt.subplot(221)
state_counts = df['state'].value_counts().head(10)
plt.pie(state_counts, labels=state_counts.index, autopct='%1.1f%%', colors=colors)
plt.title('Distribution of Top 10 States')


plt.subplot(222)
pets_counts = df['pets_allowed'].value_counts()
plt.pie(pets_counts, labels=pets_counts.index, autopct='%1.1f%%', colors=colors[:len(pets_counts)])
plt.title('Distribution of Pets Allowed')


plt.subplot(223)
photo_counts = df['has_photo'].value_counts()
plt.pie(photo_counts, labels=photo_counts.index, autopct='%1.1f%%', colors=colors[:len(photo_counts)])
plt.title('Distribution of Photo Availability')


plt.subplot(224)
fee_counts = df['fee'].value_counts()
plt.pie(fee_counts, labels=fee_counts.index, autopct='%1.1f%%', colors=colors[:len(fee_counts)])
plt.title('Distribution of Fee')

plt.tight_layout()
plt.show()

#######for amenities


# Split the amenities column and count each unique amenity
amenities_list = apts['amenities'].str.split(',').sum()
amenities_count = Counter(amenities_list).most_common(15)  # Top 15 amenities

#% Create a bar plot
amenities_df = pd.DataFrame(amenities_count, columns=['Amenity', 'Count'])
plt.figure(figsize=(10, 6))
sns.barplot(y='Amenity', x='Count', data=amenities_df, palette='bright')
plt.title("Top 15 Amenities")
plt.xlabel("Count")
plt.ylabel("Amenity")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.pie(amenities_df['Count'], labels=amenities_df['Amenity'],
        autopct='%1.1f%%',)
plt.title("Top 15 Amenities Pie Plot")
plt.tight_layout()
plt.show()

#%%
###############################
#Outlier Detection and Removal
#############################

#multivariate outlier
#using IQR outlier removal on price
numerical_cols = ['bathrooms', 'bedrooms', 'price', 'square_feet']
initial_count = apts.shape[0]
Q1 = apts[numerical_cols].quantile(0.25)
Q3 = apts[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
lower_quantile_range = Q1-1.5*IQR
upper_quantile_range = Q3+1.5*IQR

apts = apts[~((apts[numerical_cols] < lower_quantile_range) | (apts[numerical_cols] > upper_quantile_range)).any(axis=1)]
print(f"Apartment Dataset removed observation count: {initial_count-apts.shape[0]} ")
print(f"Upper Quantile Range of Features:\n{upper_quantile_range}")
print(f"Lower Quantile Range of Features:\n{lower_quantile_range}")

##Compare results after outlier removal
fig, axs = plt.subplots(4,2, figsize=(12,8))
axs = axs.flatten()

for i,col in enumerate(numerical_cols):
    #histogram plot
    sns.histplot(apts[col], kde=True, bins=30, color='blue',
                 ax=axs[2*i])
    axs[2*i].set_title(f"Histogram plot of {col} after removing outliers")
    axs[2*i].set_xlabel(col)
    axs[2*i].set_ylabel('Frequency')

    #box plot
    sns.boxplot(x=apts[col], color='orange', ax=axs[2* i+1], palette=color_palette("pastel"))
    axs[2* i+1].set_title(f"Boxplot of {col} after removing outliers")
    axs[2* i+1].set_xlabel(col)
    axs[2* i+1].set_ylabel('Frequency')
    fig.suptitle(f'Histogram and box plot after removing outliers')

plt.tight_layout()
plt.show()

#%%
from statsmodels. graphics.gofplots import qqplot
############################
#####Normality test
############################
#before
numerical_cols = ['bathrooms', 'bedrooms', 'price', 'square_feet']

plt.figure()

sns.lineplot(apts[:1000], y = 'square_feet',x = apts[:1000].index, label = 'square_feet'
             )
plt.title(f'Checking Normal Line Plot[raw data]')
plt.xlabel('Observations')
plt.ylabel('Square Feet Feature')
plt.tight_layout()
plt.show()

plt.figure()

sns.lineplot(apts[:1000], y = 'price',x = apts[:1000].index, label = 'price'
             )
plt.title(f'Checking Normal Line Plot[raw data]')
plt.xlabel('Observations')
plt.ylabel('Price')
plt.tight_layout()
plt.show()

for col in numerical_cols:
    plt.figure()
    qq_plot = qqplot(apts[col], linewidth=3, line='s',
                     label=col, markerfacecolor='blue',
                     alpha=0.5)
    plt.title(f'Q-Q Plot of {col}')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.tight_layout()
    plt.show()

###########KS Test
from scipy.stats import kstest

def ks_test(df, columns):
    z = (df[columns] - df[columns].mean())/df[columns].std()
    ks_stat, p_value = kstest(z, 'norm')
    print('='*50)
    print(f'K-S test: {columns} dataset: statistics= {ks_stat:.2f} p-value = {p_value:.2f}' )

    alpha = 0.01
    if p_value > alpha :
        print(f'K-S test:  {columns}  dataset is Normal')
    else:
        print(f'K-S test : {columns}  dataset is Not Normal')
    print('=' * 50)

#### shapiro test
from scipy.stats import shapiro

def shapiro_test(df, column):
    stats, p = shapiro(df[column])
    print('=' * 50)
    print(f'Shapiro test : {column} dataset : statistics = {stats:.2f} p-vlaue of ={p:.2f}' )
    alpha = 0.01
    if p > alpha :
        print(f'Shapiro test: {column} dataset is Normal')
    else:
        print(f'Shapiro test: {column} dataset is NOT Normal')
    print('=' * 50)

for col in numerical_cols:
    ks_test(apts, col)
    shapiro_test(apts, col)


#%% Statistics Analysis
plt.figure(figsize=(8, 6))
plt.hexbin(x=apts['square_feet'], y=apts['price'], gridsize=30, cmap='Blues', mincnt=1)
plt.colorbar(label='Count')
plt.title("Price vs Square Feet (Density)")
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(x='square_feet', y='price', data=apts, bins=30, cbar=True, cmap="YlGnBu")
plt.title("Heatmap of Price vs Square Feet")
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.show()

#state analysis
state_apts = apts.groupby('state')

state_means = state_apts['price'].mean()
state_means = state_means.reset_index()
state_means = state_means.sort_values(['price'], ascending=False)

state_counts = apts.state.value_counts()
state_counts = state_counts.reset_index()

print(state_means.reset_index())
print(state_counts)
fig, ax1 = plt.subplots(figsize=(15,8))

sns.barplot(x = 'state', y = 'price',data = state_means, ax=ax1,
            palette=sns.color_palette('bright', n_colors=3))
ax1.set_ylabel('Average Price (Bars)')

ax2 = ax1.twinx()

sns.scatterplot(x = 'state', y = 'count', data = state_counts,  ax=ax2)
ax2.set_ylabel('Number of Datapoints (Points)')

plt.show()

sns.set(font_scale=1,
palette = 'bright',
style = 'whitegrid',
)
bathrooms = apts.groupby('state', as_index=False)['bathrooms'].mean().sort_values('bathrooms')
bedrooms = apts.groupby('state', as_index=False)['bedrooms'].mean().sort_values('bedrooms')

bedrooms_with_price = bedrooms.merge(state_means, how='inner', on='state')
bedrooms_bathrooms_price = bedrooms_with_price.merge(bathrooms, how='inner', on='state')


### Analysis based on number of bedrooms and bathrooms
sns.set(font_scale=1,
palette = 'bright',
style = 'whitegrid',
)
bathrooms = apts.groupby('state', as_index=False)['bathrooms'].mean().sort_values('bathrooms')
bedrooms = apts.groupby('state', as_index=False)['bedrooms'].mean().sort_values('bedrooms')

bedrooms_with_price = bedrooms.merge(state_means, how='inner', on='state')
bedrooms_bathrooms_price = bedrooms_with_price.merge(bathrooms, how='inner', on='state')

fig, ax1 = plt.subplots(figsize=(16,6))


ax2 = ax1.twinx()
sns.barplot(x = 'state', y='bedrooms', data=bedrooms_bathrooms_price, ax=ax1,
            palette=sns.color_palette('Paired', n_colors=3))
ax1.set_ylabel('Average Bathrooms [marker], Average Bedrooms(Bars)')
sns.scatterplot(x = 'state', y = 'bathrooms',data = bedrooms_bathrooms_price,  ax=ax1,
                palette=sns.color_palette('bright', n_colors=3))

sns.lineplot(x = 'state', y = 'price',data = bedrooms_bathrooms_price, ax=ax2,
             palette=sns.color_palette('bright', n_colors=3))
ax2.set_ylabel('Average Price (Line)')
plt.tight_layout()
plt.show()

##square feet analysis
sqft_apts = apts.groupby('state', as_index=False)['square_feet'].mean()

sqft_apts = sqft_apts.merge(state_means, how='inner', on='state')

sqft_apts['dollar_per_sqft'] = sqft_apts['price']/sqft_apts['square_feet']
sqft_apts = sqft_apts.sort_values('dollar_per_sqft')

fig, ax1 = plt.subplots(figsize=(16,6))
sns.barplot(x = 'state', y='dollar_per_sqft', data=sqft_apts,
            palette=sns.color_palette('Paired', n_colors=3))
ax1.set_ylabel('Dollars Per Square Foot [bars]')

ax2 = ax1.twinx()
sns.scatterplot(x = 'state', y = 'square_feet',data = sqft_apts,  ax=ax2,
                palette=sns.color_palette('Paired', n_colors=3))
ax2.set_ylabel('Average Square Feet [markers]')

plt.show()

#%%
##########################
##Correlation Analysis
##########################
correlation = apts[numerical_cols].corr()
covariance_plots = apts[numerical_cols].cov()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(covariance_plots, annot=True, cmap='Paired', fmt=".2f")
plt.title("Covariance Heatmap")
plt.show()

sns.pairplot(apts[numerical_cols], diag_kind='kde', plot_kws={'alpha': 0.7},
             palette=sns.color_palette('coolwarm', n_colors=5))
plt.show()

#%%%%
##################
#Clustering for latitude and longitude analysis
###################
#just for analysis later will be doing clustering in better way
# Clustering
coords = apts[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=5, random_state=42).fit(coords)
apts['cluster'] = kmeans.labels_

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='longitude', y='latitude', hue='cluster', palette='tab10', data=apts, alpha=0.6)
plt.title("Apartment Locations (Clustered)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title='Cluster')
plt.show()
apts.drop(columns = 'cluster',axis=1, inplace=True)#created only for analysis

#%%
#################################
#One hot encoding
##################################
def convert_categorical_data(apts, col, remove):
    apts[col] = apts[col].str.strip()
    apts = pd.concat([apts, apts[col].str.get_dummies(sep=',')], axis=1)
    if remove == True:
        apts = apts.drop([col], axis=1)
    return apts


apts = convert_categorical_data(apts, 'amenities', True)
#
print(f"All columns after one hot encoding: {apts.columns.tolist()}")
print(f"Number of columns after one hot encoding of amenities feature: {apts.shape[1]}")



#%% Added extra columns like- 'address', 'neighborhood','county','postcode', 'place_importance','place_rank
#using latitude and longitude features in dataset using geopy api, for getting better details which helps in predicting price
#since it took a lot of time processing I have created final csv file with complete data, you can run below code
#to check the results I obtained you can run the below code....
#######################
#Latitude and Longitude details fetch
#######################
# from tqdm import tqdm
# import time
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter
# from multiprocessing import Pool, cpu_count
#
# apts = apts[0:10000]
# geolocator = Nominatim(user_agent="large_geocoding", timeout=10)
# geocode = RateLimiter(geolocator.reverse,  max_retries=3)
#
#
# def geocode_location(row):
#
#     try:
#         location = geocode((row['latitude'], row['longitude']))
#         if location:
#             address = location.raw.get('address', {})
#             return {
#                 'address': location.address,
#                 'neighborhood': address.get('neighbourhood'),
#                 'county': address.get('county'),
#                 'postcode': address.get('postcode'),
#                 'place_importance': location.raw.get('importance'),
#                 'place_rank': location.raw.get('place_rank'),
#             }
#     except Exception as e:
#         print(f"Error geocoding latitude {row['latitude']}, longitude {row['longitude']}: {e}")
#     return {
#         'address': None,
#         'neighborhood': None,
#         'county': None,
#         'postcode': None,
#         'place_importance': None,
#         'place_rank': None,
#     }
#
#
# def parallel_geocode(df, chunk_size=100):
#
#     start_time = time.time()
#
# #parallel processing
#     chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
#
#
#     results = []
#     with Pool(processes=cpu_count()) as pool:
#         for result in tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="Geocoding Progress"):
#             results.extend(result)
#
#     elapsed_time = time.time() - start_time
#     print(f"Geocoding completed in {elapsed_time:.2f} seconds.")
#     return pd.DataFrame(results)
#
#
# def process_chunk(chunk):
#     return [geocode_location(row) for _, row in chunk.iterrows()]
#
# print("Starting geocoding process...")
# geocoded_data = parallel_geocode(apts)
#
#
# result_df = pd.concat([apts.reset_index(drop=True), geocoded_data], axis=1)
# result_df.to_csv("geocoded_data_1.csv", index=False)

###############processed for 10k records and combined them in one csv file########
#% combining all records with rank, importance and address extracted from latitude and longitude features
# import pandas as pd
# import requests
# from io import StringIO
# import os
# def combine_github_csvs(github_urls, output_file):
#
#     df_list = []
#
#     for url in github_urls:
#         try:
#             response = requests.get(url)
#             response.raise_for_status()
#
#             csv_content = StringIO(response.text)
#             df = pd.read_csv(csv_content)
#
#             df_list.append(df)
#             print(f"Successfully processed {url}: {len(df)} rows")
#
#         except Exception as e:
#             print(f"Error processing {url}: {str(e)}")
#             continue
#
#     if df_list:
#         combined_df = pd.concat(df_list, ignore_index=True)
#         combined_df.to_csv(output_file, index=False)
#         print(f"\nCombined CSV created successfully!")
#         print(f"Total rows: {len(combined_df)}")
#     else:
#         print("No data was successfully downloaded and processed.")
#
# github_urls = [
#     "https://raw.githubusercontent.com/KaurHarleen1930/apartment_rent-private-repo/refs/heads/main/ML/geocoded_data_1.csv",
#     "https://raw.githubusercontent.com/KaurHarleen1930/apartment_rent-private-repo/refs/heads/main/ML/geocoded_data_2.csv",
#     "https://raw.githubusercontent.com/KaurHarleen1930/apartment_rent-private-repo/refs/heads/main/ML/geocoded_data_3.csv",
#     "https://raw.githubusercontent.com/KaurHarleen1930/apartment_rent-private-repo/refs/heads/main/ML/geocoded_data_4.csv",
#     "https://raw.githubusercontent.com/KaurHarleen1930/apartment_rent-private-repo/refs/heads/main/ML/geocoded_data_5.csv",
#     "https://raw.githubusercontent.com/KaurHarleen1930/apartment_rent-private-repo/refs/heads/main/ML/geocoded_data_6.csv",
#     "https://raw.githubusercontent.com/KaurHarleen1930/apartment_rent-private-repo/refs/heads/main/ML/geocoded_data_7.csv",
#     "https://raw.githubusercontent.com/KaurHarleen1930/apartment_rent-private-repo/refs/heads/main/ML/geocoded_data_8.csv",
#     "https://raw.githubusercontent.com/KaurHarleen1930/apartment_rent-private-repo/refs/heads/main/ML/geocoded_data_9.csv"
# ]
#
# output_file = "combined_geocoded_data.csv"
# combine_github_csvs(github_urls, output_file)
##########################################created a final file merging all data with new fields ####################
#%% using combined data going forward - loading data and cleaning newly added features
url = "https://raw.githubusercontent.com/KaurHarleen1930/apartment_rent-private-repo/refs/heads/main/ML/combined_geocoded_data.csv"
#######please uncomment line 792 and comment line 790 if you want to read data locally
#url = 'combined_geocoded_data.csv'
apts_geocoded = pd.read_csv(url)

print(f"Shape of apts: {apts_geocoded.shape}")
print(f"First five rows: {apts_geocoded.head()}")
print(f"Null values: {apts_geocoded.isnull().sum()}")
print(f"Duplicated: {apts_geocoded.duplicated().sum()}")
####checking for any null values in new columns added
apts_geocoded["pets_allowed"] = apts_geocoded["pets_allowed"].fillna("None")
print(f"Null values: {apts_geocoded.isnull().sum()}")
###dropped neighborhood column as it was one of newly added columns
# after fetching details from latitude and longitude columns but majorly values were null so deleted it....
apts_geocoded.drop(axis=1, columns='neighborhood', inplace=True)
#######other columns has few rows null, so dropped null values
apts_geocoded.dropna(axis=0, how='any', inplace=True)
print(f"Shape of apts: {apts_geocoded.shape}")
print(f"First five rows: {apts_geocoded.head()}")
print(f"Null values: {apts_geocoded.isnull().sum()}")

#%%
##########################################################
###One hot encoding and label encoding for other categorical features
#################################################################

apts_geocoded['fee'] = apts_geocoded['fee'].map({'Yes':1, 'No':0})
apts_geocoded['has_photo'] = apts_geocoded['has_photo'].map({'Yes':2, 'Thumbnail':1,'No':0})

apts_geocoded['pets_allowed'] = apts_geocoded['pets_allowed'].str.strip()
dummies_pets = apts_geocoded['pets_allowed'].str.get_dummies(sep=',')
dummies_pets = pd.DataFrame(dummies_pets)
dummies_pets.rename(columns = {'Cats':'Pet_Cats','Dogs':'Pet_Dogs','None':'Pet_None'}, inplace = True)
apts_geocoded = pd.concat([apts_geocoded, dummies_pets], axis=1)
apts_geocoded.drop(inplace = True, axis = 1, columns='pets_allowed')

#################################################################
###Mean Price Encoding for cityname feature

# Maps each city to the average price of properties in that city
# Uses smoothing to handle cities with few properties
def encode_cityname(df):
    df_encoded = df.copy()
    city_price_stats = df.groupby('cityname').agg({
        'price': ['mean', 'count']
    }).reset_index()

    global_mean = df['price'].mean()
    smoothing_factor =int((df.shape[0]/len(df['cityname'].value_counts()))*0.3)

    city_price_stats['smoothed_mean'] = (
            (city_price_stats[('price', 'mean')] * city_price_stats[('price', 'count')] +
             global_mean * smoothing_factor) /
            (city_price_stats[('price', 'count')] + smoothing_factor)
    )

    df_encoded['cityname_mean_price'] = df_encoded['cityname'].map(
        dict(zip(city_price_stats['cityname'], city_price_stats['smoothed_mean']))
    )

    #################################################################
    ### Price Rank Encoding for cityname feature
    city_rank = df.groupby('cityname')['price'].mean().rank(method='dense')
    df_encoded['cityname_price_rank'] = df_encoded['cityname'].map(city_rank)
    return df_encoded


df_encoded = encode_cityname(apts_geocoded)
df_encoded.drop(columns='cityname', inplace=True)


##########################
#Mean price for State column encoding
############################
def encode_state(df):
    df_encoded = df.copy()
    state_price_stats = df.groupby('state').agg({
        'price': ['mean', 'count']
    }).reset_index()
    global_mean = df['price'].mean()
    smoothing_factor =int((df.shape[0]/len(df['state']))*0.2)
    state_price_stats['smoothed_mean'] = (
            (state_price_stats[('price', 'mean')] * state_price_stats[('price', 'count')] +
             global_mean * smoothing_factor) /
            (state_price_stats[('price', 'count')] + smoothing_factor)
    )
    df_encoded['state_mean_price'] = df_encoded['state'].map(
        dict(zip(state_price_stats['state'], state_price_stats['smoothed_mean']))
    )
    return df_encoded

df_encoded_state = encode_state(df_encoded)
df_encoded_state.drop(columns='state', inplace=True)

################################
##Postcode encoding
################################
def encode_postcode(df):
    df_encoded = df.copy()
    global_mean = df['price'].mean()
    postcode_stats = df.groupby('postcode').agg({
        'price': ['mean', 'count']
    })

    # Calculate postcode smoothing factor (35% - higher due to more granularity)
    avg_samples_per_postcode = len(df) / len(df['postcode'].unique())
    postcode_smoothing = int(avg_samples_per_postcode * 0.35)

    # Apply smoothing for postcodes
    postcode_smoothed_means = (
            (postcode_stats[('price', 'mean')] * postcode_stats[('price', 'count')] +
             global_mean * postcode_smoothing) /
            (postcode_stats[('price', 'count')] + postcode_smoothing)
    )
    df_encoded['postcode_encoded'] = df['postcode'].map(postcode_smoothed_means)
    return df_encoded

encoded_features = encode_postcode(df_encoded_state)
encoded_features.drop(columns='postcode', inplace=True)

print(encoded_features[['postcode_encoded','state_mean_price','cityname_price_rank','cityname_mean_price', 'price']].head())
#%%
###############################################################
#Analyxing correlation between location features added newly
##########################################################
def analyze_location_features(df):
    # Select relevant features
    location_features = [
        'cityname_mean_price',
        'postcode_encoded',
        'state_mean_price',
        'place_importance',
        'place_rank',
        'latitude',
        'longitude',
        'price'  # Included price to see relationships with target
    ]

    corr_matrix = df[location_features].corr()

    # Spearman correlation (better for potential non-linear relationships)
    spearman_corr = pd.DataFrame(
        [[spearmanr(df[col1], df[col2])[0] for col1 in location_features]
         for col2 in location_features],
        columns=location_features,
        index=location_features
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                ax=ax1)
    ax1.set_title('Pearson Correlation')

    sns.heatmap(spearman_corr,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                ax=ax2)
    ax2.set_title('Spearman Correlation')

    plt.tight_layout()
    plt.show()

    print("\nFeature Correlations with Price:")
    price_correlations = abs(spearman_corr['price']).sort_values(ascending=False)
    for feature, corr in price_correlations.items():
        if feature != 'price':
            print(f"{feature:20} | Correlation: {corr:.3f}")


    print("\nPotentially Redundant Features (|correlation| > 0.7):")
    for i in range(len(location_features)):
        for j in range(i + 1, len(location_features)):
            if abs(spearman_corr.iloc[i, j]) > 0.7:
                print(
                    f"{location_features[i]:20} - {location_features[j]:20} | Correlation: {spearman_corr.iloc[i, j]:.3f}")

    return corr_matrix, spearman_corr

def analyze_place_metrics(df):

    print("\nPlace Metrics Analysis:")

    # Basic statistics
    print("\nplace_importance statistics:")
    print(df['place_importance'].describe())

    print("\nplace_rank statistics:")
    print(df['place_rank'].describe())


corr_matrix, spearman_corr = analyze_location_features(encoded_features)
analyze_place_metrics(encoded_features)

#%%

###Creating final dataframe with all the required features and converted after encodings
final_df = pd.DataFrame(encoded_features, columns=['bathrooms', 'bedrooms', 'fee', 'has_photo', 'price',
       'square_feet','AC', 'Alarm', 'Basketball', 'Cable or Satellite', 'Clubhouse',
       'Dishwasher', 'Doorman', 'Elevator', 'Fireplace', 'Garbage Disposal',
       'Gated', 'Golf', 'Gym', 'Hot Tub', 'Internet Access', 'Luxury',
       'Parking', 'Patio/Deck', 'Playground', 'Pool', 'Refrigerator',
       'Storage', 'TV', 'Tennis', 'View', 'Washer Dryer', 'Wood Floors',
       'no amenities available','place_importance', 'place_rank', 'Pet_Cats', 'Pet_Dogs', 'Pet_None',
       'cityname_mean_price', 'cityname_price_rank', 'state_mean_price',
       'postcode_encoded'])

##removed lattude and longitude as already extracted required info

#%%
#############################################################################
#Standarizing the dataset
#############################################################################
scaler = StandardScaler()
features_to_scale = [
    'bathrooms',
    'bedrooms',
    'square_feet',
    'cityname_mean_price',
    'cityname_price_rank',
    'state_mean_price',
    'postcode_encoded',
    'place_importance'
]
final_df[features_to_scale] = scaler.fit_transform(final_df[features_to_scale])

#%% Prepare Data
###############################
#Split Data
##############################
def prepare_data(df, target = 'price'):
    df['price_category'] = pd.qcut(df[target], q=3, labels=['low', 'medium', 'high'])
    label_mapping = {'low': 0, 'medium': 1, 'high': 2}
    df['price_category'] = df['price_category'].map(label_mapping)

    X = df.drop(columns=[target, 'price_category'], axis=1)
    y_reg = df[target]
    y_clf = df['price_category']

    #split data
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=5805,
        stratify=y_clf
    )
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test
#%%
#############################################################################
#Dimensionality reduction/feature selection
#############################################################################
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = prepare_data(final_df)

def dimensionality_reduction(X,y):
    #Correlation Analysis
    pd.set_option('display.max_columns', None)
    print("Correlation Analysis")
    print("="*50)
    correlation_matrix = X.corr()
    covariance_matrix = X.cov()

    high_corr_features = np.where(np.abs(correlation_matrix) > 0.7)
    high_corr_features = [(correlation_matrix.index[x], correlation_matrix.columns[y],
                           correlation_matrix.iloc[x, y])
                          for x, y in zip(*high_corr_features) if x != y and x < y]

    if high_corr_features:
        print("\nHighly correlated feature pairs (>0.7):")
        for feat1, feat2, corr in high_corr_features:
            print(f"{feat1} - {feat2}: {corr:.3f}")

    #Random Forest
    print("Random Forest Analysis")
    print("=" * 50)
    model = RandomForestRegressor(random_state=5805)
    model.fit(X, y)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.show()

    # PCA Analysis
    print("PCA Analysis")
    print("=" * 50)
    pca = PCA()
    pca.fit_transform(X)
    explained_variance = pd.DataFrame({
        'Component': range(1, len(pca.explained_variance_ratio_) + 1),
        'Explained_Variance': pca.explained_variance_ratio_,
        'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
    })
    cumm_variance = np.cumsum(pca.explained_variance_ratio_)
    features_95_threshold = np.argmax(cumm_variance >= 0.95) + 1
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(explained_variance['Component'],
             explained_variance['Explained_Variance'], 'bo-')
    plt.title('Explained Variance Ratio Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')

    plt.subplot(1, 2, 2)
    plt.plot(explained_variance['Component'],
             explained_variance['Cumulative_Variance'], 'ro-')
    plt.axhline(y=0.95, color='k', linestyle='--')
    plt.axvline(x = features_95_threshold, color='k', linestyle='--')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.tight_layout()
    plt.show()

    condition_number = np.linalg.cond(X)
    print(f"\nCondition Number: {condition_number:.2f}")

    #SVD
    print("\n4. SVD Analysis")
    print("=" * 50)
    n_components = min(X.shape[1] - 1, 100)  # Adjust number of components
    svd = TruncatedSVD(n_components=n_components)
    svd_result = svd.fit_transform(X)
    explained_variance_ratio = svd.explained_variance_ratio_
    print("\nSingular Values:")
    for i, singular_value in enumerate(svd.singular_values_, 1):
        print(f"Component {i}: {singular_value:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(svd.singular_values_) + 1),
             svd.singular_values_, 'go-')
    plt.title('Singular Values')
    plt.xlabel('Component')
    plt.ylabel('Singular Value')
    plt.tight_layout()
    plt.show()

    #VIF Analysis
    print("\n5. VIF Analysis")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X, i)
                       for i in range(X.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False)

    print("\nVariance Inflation Factors:")
    print(vif_data)

    ###finding features to remove todo where feature importance is zero
    features_to_remove = set()

    remove_non_important = feature_importance[feature_importance['importance'] < 0.001]['feature'].tolist()
    for feat in remove_non_important:
        features_to_remove.add(feat)

    vif_features_remove = vif_data[vif_data['VIF'] > 5]['Feature'].tolist()
    features_to_remove.update(vif_features_remove)
    ###pca 95%variance
    n_components_95 = np.argmax(explained_variance['Cumulative_Variance'] >= 0.95) + 1

    print("\nDimensionality Reduction Summary:")
    print(f"Original number of features: {X.shape[1]}")
    print(f"Number of highly correlated feature pairs: {len(high_corr_features)}")
    print(f"Features with high VIF (>5): {len(vif_features_remove)}")
    print(f"PCs needed for 95% variance explained: {n_components_95}")
    print(f"\n features importance: {feature_importance}")
    print(f"\nRecommended features to remove: {sorted(features_to_remove)}")


    return correlation_matrix, feature_importance,explained_variance,vif_data,features_to_remove


correlation_matrix, rf_importance,pca_results,vif_data,features_to_remove = dimensionality_reduction(X_train, y_reg_train)
#print(f"Features to remove: {features_to_remove}")

#%%
#####################
##Removing features
#####################
####Result from analyzing to remove features- ['Pet_Cats', 'Pet_Dogs', 'Pet_None', 'bathrooms', 'cityname_mean_price', 'cityname_price_rank', 'has_photo', 'place_rank', 'state_mean_price']
#'cityname_price_rank', 'has_photo', 'bathrooms', 'state_mean_price'
final_df.drop(columns = ['Alarm', 'Basketball', 'Doorman', 'Garbage Disposal', 'Golf', 'Hot Tub', 'Luxury', 'Pet_Cats', 'Pet_Dogs', 'Pet_None', 'View', 'cityname_mean_price', 'fee', 'place_rank'], axis = 1, inplace = True)
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = prepare_data(final_df)
#after
correlation_matrix, rf_importance,pca_results,vif_data,features_to_remove = dimensionality_reduction(X_train, y_reg_train)






#%%
print(f"Final Selected columns: {final_df.columns}")
print(f"Total Number of columns selected: {len(final_df.columns)}")

#%%
############################################################################
######################## PHASE 2 - Regression Analysis###########################################
############################################################################

def displayTable(reg_results):
    results_df = pd.DataFrame(reg_results,
                              columns=["Model", "Root Mean Squared Error", "Mean Absolute Error", "R2 Score",
                                       "R2 Adjusted"])
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    result_table = PrettyTable()
    result_table.field_names = results_df.columns
    for row in results_df.itertuples(index=False):
        result_table.add_row(row)

    print(result_table)

def evaluate_regression_models(X_train, X_test, y_reg_train, y_reg_test):
    models= {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=5805),
        'Random Forest': RandomForestRegressor(random_state=5805),
        'SVR': SVR(),
        'Neural Network': MLPRegressor(random_state=5805)
    }
    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_reg_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_reg_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_reg_test, y_pred)
        r2 = r2_score(y_reg_test, y_pred)

        results.append({
            'Model': name,
            'Root Mean Squared Error': rmse,
            'Mean Absolute Error': mae,
            'R2 Score': r2,
            'R2 Adjusted': ''
        })

        print(f"{name} Results:")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"R2 Score: {r2:.4f}")

        ################################subplots####################################
        residuals = y_reg_test - y_pred

        # Subplot for each model
        fig = plt.figure(figsize=(20, 12))

        plt.subplot(2, 3, 1)
        plt.scatter(y_reg_test, y_pred, alpha=0.5)
        plt.plot([y_reg_test.min(), y_reg_test.max()],
                 [y_reg_test.min(), y_reg_test.max()],
                 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'Actual vs Predicted for model:{name}')

        plt.subplot(2, 3, 2)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title(f'Residuals vs Predicted for model: {name}')

        plt.subplot(2, 3, 3)
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residual Value')
        plt.ylabel('Count')
        plt.title(f'Distribution of Residuals for model: {name}')

        plt.subplot(2, 3, 4)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot')

        plt.subplot(2, 3, 5)
        plt.plot(residuals, marker='o', linestyle='none', alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Index')
        plt.ylabel('Residuals')
        plt.title(f'Residuals vs Index for model: {name}')

        plt.subplot(2, 3, 6)
        error_percent = (residuals / y_reg_test) * 100
        sns.histplot(error_percent, kde=True)
        plt.xlabel('Percentage Error')
        plt.ylabel('Count')
        plt.title(f'Distribution of Percentage Errors for model: {name}')

        plt.tight_layout()
        plt.show()

    return results, models

def run_regression_model(df):
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = prepare_data(df)
    print("Running Regression Models...")
    reg_results, reg_models = evaluate_regression_models(
        X_train, X_test, y_reg_train, y_reg_test
    )

    return reg_results, reg_models

reg_results, reg_models = run_regression_model(final_df)
print("\nRegression Models Summary:")
displayTable(reg_results)
#%%
#################################
##Using backward elimination to decrease number of features
##################################
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = prepare_data(final_df)
observations_train = X_train
observations_test = X_test
new_data = observations_train.copy()
features = new_data.columns.tolist()
remove_features = []
table_summary = []
prev_r2 = 0
prev_col = ''
#if p-value > threshold then fail to reject null hypothesis - no relationship between X and Y
while(len(features)>0):
    model = sm.OLS(y_reg_train, observations_train[features]).fit()
    aic = model.aic.round(3)
    bic = model.bic.round(3)

    adjusted_r2 = model.rsquared_adj.round(3)
    p_values = pd.Series(model.pvalues.round(3), index = new_data[features].columns)
    p_values_max = p_values.max()
    if p_values.max() > 0.01:
        col = p_values.idxmax()
        features.remove(col)
        remove_features.append(col)
        table_summary.append({
            "Feature": col,
            "AIC": aic,
            "BIC": bic,
            "Adjusted R2": adjusted_r2,
            "p-value": p_values_max
        })
        prev_r2 = adjusted_r2
        prev_col = col
    else:
        if(adjusted_r2 < prev_r2):
            features.append(prev_col)
            remove_features.remove(prev_col)
        print(model.summary())
        break

    print(model.summary())


result_df = pd.DataFrame(table_summary, columns=["Feature", "AIC", "BIC", "Adjusted R2", "p-value"])
result_table = PrettyTable()
result_table.field_names = result_df.columns
for row in result_df.itertuples(index=False):
    result_table.add_row(row)
print(result_table)
print(f'Final features after stepwise regression: {features}')
print(f'Remove Features: {remove_features}')

#%%
#################################
##Dropping features after backward elimination
##################################
final_df.drop(columns = remove_features, inplace = True)
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = prepare_data(final_df)
#%%
###########################
##OLS Regression analysis
###########################
def create_stats_table(model):
    t_table = PrettyTable()
    t_table.field_names = ["Feature", "T-statistic", "P-value", "Significant"]

    for feature, t_val, p_val in zip(model.model.exog_names,model.tvalues,model.pvalues):
        t_table.add_row([
            feature,
            f"{t_val:.3f}",
            f"{p_val:.3f}",
            "Yes" if p_val < 0.01 else "No"
        ])
    print("T-test Analysis Results:")
    print(t_table)


def create_performance_table(metrics_dict):
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Value"]
    for metric, value in metrics_dict.items():
        perf_table.add_row([metric, f"{value:.3f}"])

    print("\nModel Performance Summary:")
    print(perf_table)


def create_ci_table(model):
    ci_table = PrettyTable()
    ci_table.field_names = ["Feature", "Coefficient", "CI Lower", "CI Upper"]
    conf_int = model.conf_int()
    for feature, coef, ci_l, ci_u in zip(model.model.exog_names,model.params,conf_int[0],conf_int[1]):
        ci_table.add_row([
            feature,
            f"{coef:.3f}",
            f"{ci_l:.3f}",
            f"{ci_u:.3f}"
        ])

    print("\nConfidence Intervals (95%):")
    print(ci_table)
    ci_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.params,
        'CI_Lower': conf_int[0],
        'CI_Upper': conf_int[1]
    })
    ci_df = ci_df.reindex(ci_df.Coefficient.abs().sort_values(ascending=True).index)
    plt.figure(figsize=(12, 8))

    y_pos = np.arange(len(ci_df))

    plt.hlines(y=y_pos, xmin=ci_df['CI_Lower'], xmax=ci_df['CI_Upper'],
               color='skyblue', alpha=0.5, linewidth=5)

    plt.plot(ci_df['Coefficient'], y_pos, 'o', color='navy',
             markersize=8, label='Coefficient')

    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    plt.ylabel('Features')
    plt.xlabel('Coefficient Value')
    plt.title('95% Confidence Intervals for Model Coefficients')
    plt.yticks(y_pos, ci_df['Feature'])

    # Add grid for better readability
    plt.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

def show_subplots_reg(y_reg_test,y_pred):
    residuals = y_reg_test - y_pred
    name = 'Statsmodels OLS'
    fig = plt.figure(figsize=(20, 12))

    plt.subplot(2, 3, 1)
    plt.scatter(y_reg_test, y_pred, alpha=0.5)
    plt.plot([y_reg_test.min(), y_reg_test.max()],
             [y_reg_test.min(), y_reg_test.max()],
             'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Actual vs Predicted for model:{name}')

    plt.subplot(2, 3, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted for model: {name}')

    plt.subplot(2, 3, 3)
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residual Value')
    plt.ylabel('Count')
    plt.title(f'Distribution of Residuals for model: {name}')

    plt.subplot(2, 3, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')

    plt.subplot(2, 3, 5)
    plt.plot(residuals, marker='o', linestyle='none', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Index for model: {name}')

    plt.subplot(2, 3, 6)
    error_percent = (residuals / y_reg_test) * 100
    sns.histplot(error_percent, kde=True)
    plt.xlabel('Percentage Error')
    plt.ylabel('Count')
    plt.title(f'Distribution of Percentage Errors for model: {name}')

    plt.tight_layout()
    plt.show()


def regression_analysis(X_train, X_test, y_reg_train, y_reg_test):
    ols_regression_model = sm.OLS(y_reg_train, X_train).fit()
    print("\nOLS Regression Summary:")
    print(ols_regression_model.summary())
    create_stats_table(ols_regression_model)
    f_table = PrettyTable()
    f_table.field_names = ["Metric", "Value"]
    f_table.add_row(["F-statistic", f"{ols_regression_model.fvalue:.4f}"])
    f_table.add_row(["F-test p-value", f"{ols_regression_model.f_pvalue:.4f}"])
    print("\nF-test Results:")
    print(f_table)
    create_ci_table(ols_regression_model)
    metrics = {
        'R-squared': ols_regression_model.rsquared,
        'Adjusted R-squared': ols_regression_model.rsquared_adj,
        'AIC': ols_regression_model.aic,
        'BIC': ols_regression_model.bic
    }
    create_performance_table(metrics)
    y_pred = ols_regression_model.predict(X_test)
    residuals = y_reg_test - y_pred
    ########MSE#########
    mse_sm = ols_regression_model.mse_resid
    rmse_sm = np.sqrt(mse_sm)
    mae_sm = mean_absolute_error(y_reg_test, y_pred)
    r2_sm = ols_regression_model.rsquared
    r2_sm_adj = ols_regression_model.rsquared_adj
    reg_results.append({
        'Model': 'Statsmodels OLS',
        'Root Mean Squared Error': rmse_sm,
        'Mean Absolute Error': mae_sm,
        'R2 Score': r2_sm,
        'R2 Adjusted': r2_sm_adj
    })
    displayTable(reg_results)

    #####show subplots
    show_subplots_reg(y_reg_test,y_pred)
    return {
        'model': ols_regression_model,
        'predictions': y_pred,
        'residuals': residuals
    }
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = prepare_data(final_df)
results_reg = regression_analysis(X_train, X_test, y_reg_train, y_reg_test)

#%%
######################################################################################################################
#PHASE 3 Classification
######################################################################################################################

#########################
#Decision Tree - Analysis
#########################

###finding parameters
def get_tree_parameter_ranges(X_train, y_train):

    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
#max_depth
    min_depth = int(np.floor(np.log2(n_samples)))
    max_depth = int(n_samples / 10)
    # Create range with reasonable steps
    depth_range = np.linspace(min_depth, max_depth, 5, dtype=int)

    print(f"\nSuggested max_depth range:")
    print(f"Dataset has {n_samples} samples")
    print(f"Minimum depth (log2(n_samples)): {min_depth}")
    print(f"Maximum depth (n_samples/10): {max_depth}")
    print(f"Suggested range: {depth_range}")

#min_samples_split
    min_split_min = max(2, int(n_samples * 0.001))  # at least 2
    min_split_max = int(n_samples * 0.05)
    split_range = np.linspace(min_split_min, min_split_max, 4, dtype=int)

    print(f"\nSuggested min_samples_split range:")
    print(f"Minimum (0.1% of samples): {min_split_min}")
    print(f"Maximum (5% of samples): {min_split_max}")
    print(f"Suggested range: {split_range}")

    # 3. min_samples_leaf
    # Rule of thumb: 0.05% to 2.5% of total samples
    min_leaf_min = max(1, int(n_samples * 0.0005))  # at least 1
    min_leaf_max = int(n_samples * 0.025)
    leaf_range = np.linspace(min_leaf_min, min_leaf_max, 4, dtype=int)

    print(f"\nSuggested min_samples_leaf range:")
    print(f"Minimum (0.05% of samples): {min_leaf_min}")
    print(f"Maximum (2.5% of samples): {min_leaf_max}")
    print(f"Suggested range: {leaf_range}")

    return {
        'max_depth': depth_range.tolist(),
        'min_samples_split': split_range.tolist(),
        'min_samples_leaf': leaf_range.tolist()
    }



param_ranges = get_tree_parameter_ranges(X_train, y_clf_train)
###
def decision_tree_analysis(X_train, X_test, y_train, y_test, param_ranges):
    print("Decision Tree Pruning Analysis")
    print("=" * 50)
    print("\nPre-pruning Analysis")
    print("-" * 30)
    # pre_prune_params = {
    # 'criterion': ['gini', 'entropy'],
    # 'max_depth': param_ranges['max_depth'],
    # 'min_samples_split': param_ranges['min_samples_split'],
    # 'min_samples_leaf': param_ranges['min_samples_leaf'],
    # 'max_features': ['sqrt', 'log2'],
    # 'min_impurity_decrease': [0.0001, 0.001, 0.01]
    # }
    # dt_pre_prune = DecisionTreeClassifier(random_state=5805)
    # pre_pruning_grid = GridSearchCV(
    #     dt_pre_prune,
    #     pre_prune_params,
    #     cv=5,
    #     scoring='accuracy',
    #     n_jobs=-1
    # )
    # pre_pruning_grid.fit(X_train, y_train)


    print("\nPre-pruning Best Parameters:")
#    print(pre_pruning_grid.best_params_)
##########################################
    #best results
    # {'criterion': 'entropy', 'max_depth': 1713, 'max_features': 'sqrt', 'min_impurity_decrease': 0.0001, 'min_samples_leaf': 34, 'min_samples_split': 68}

    pre_prune_decision_tree = DecisionTreeClassifier(random_state=5805,
                                                     max_depth=1713,
                                                     criterion = 'entropy',
                                                     max_features = 'sqrt',
                                                     min_impurity_decrease = 0.0001,
                                                     min_samples_leaf = 34,
                                                     min_samples_split  = 68
                                                     )
    pre_prune_decision_tree.fit(X_train, y_train)
    y_pred_pre_pruning = pre_prune_decision_tree.predict(X_test)

    ############
    ##Metrics
    ############
    accuracy_pre_pruning = accuracy_score(y_test, y_pred_pre_pruning)

    print(f'Accuracy on test set: {round(accuracy_pre_pruning, 2)}')
    # print(f'Confusion Matrix: {round(cm_preprune, 2)}')
    # print(f'Recall: {round(recall_preprune, 2)}')
    # print(f'AUC: {round(auc_preprune, 2)}')

    #
    plt.figure(figsize=(25, 15))
    columns = X_train.columns
    plot_tree(pre_prune_decision_tree,
              feature_names=columns,
              class_names=['Low', 'Medium', 'High'],
              filled=True,
              rounded=True,
              fontsize=8,
              precision=2
              )
    plt.title('Decision Tree with pre pruning', fontdict={'fontsize': 20})
    plt.tight_layout()
    plt.show()

    return pre_prune_decision_tree

pre_pruning_results_decision_tree = decision_tree_analysis(X_train, X_test, y_clf_train, y_clf_test, param_ranges)


def evaluate_tree_performance(tree, X_test, y_test, tree_name):
    y_pred = tree.predict(X_test)
    y_pred_prob = tree.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Calculate metrics for each class
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # ROC AUC score for multiclass
    auc_score = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

    print(f"\nResults for {tree_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC Score: {auc_score:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'confusion_matrix': cm
    }


pre_prune_metrics = evaluate_tree_performance(pre_pruning_results_decision_tree, 
                                           X_test, y_clf_test, "Pre-pruned Tree")
    #%%post prune
def post_prune_analysis_DT(X_train, X_test, y_train, y_test):
    print("\nPost-pruning Analysis")
    print("-" * 30)
    # dt_post_prune_model = DecisionTreeClassifier(random_state=5805)
    # path = dt_post_prune_model.cost_complexity_pruning_path(X_train, y_train)
    # ccp_alphas = path.ccp_alphas[:-1]
    # dt_scores = []
    # dt_trees = []
    # for ccp_alpha in ccp_alphas:
    #     dt = DecisionTreeClassifier(random_state=5805, ccp_alpha=ccp_alpha)
    #     dt.fit(X_train, y_train)
    #     score = dt.score(X_test, y_test)
    #     dt_scores.append(score)
    #     dt_trees.append(dt)
    #
    # best_alpha_idx = np.argmax(dt_scores)
    # best_ccp_alpha = ccp_alphas[best_alpha_idx]
    # best_post_tree = dt_trees[best_alpha_idx]
    # print(f'Optimum alpha in the cost complexity function: {best_ccp_alpha}')
    # print(f'Test score with optimal alpha: {round(dt_scores[best_alpha_idx], 2)}')
    ################################################
    #Obtained best ccp_alpha values as '1.4626826687905199e-05'----
    # To verify please un comment above code - Estimated time- 4 hours
    optimum_ccp_alpha_val = 1.4626826687905199e-05
    best_post_tree = DecisionTreeClassifier(random_state=5805, ccp_alpha=optimum_ccp_alpha_val)
    best_post_tree.fit(X_train, y_train)
    y_pred_post_pruning = best_post_tree.predict(X_test)
    ################################################################
    #performance
    ########################

    post_accuracy = accuracy_score(y_test, y_pred_post_pruning)

    print(f'Accuracy on test set: {round(post_accuracy, 2)}')
    plt.figure(figsize=(25, 15))
    columns = X_train.columns
    plot_tree(best_post_tree,
              feature_names=columns,
              class_names=['Low', 'Medium', 'High'],
              filled=True,
              rounded=True,
              fontsize=8,
              precision=2
              )
    plt.title('Decision Tree with post pruning', fontdict={'fontsize': 20})
    plt.tight_layout()
    plt.show()

    ####post plot
    # plt.figure()
    # plt.plot(ccp_alphas, dt_scores, label='test score', marker='o', drawstyle="steps-post")
    # plt.grid(True)
    # plt.title('Accuracy score of the test set')
    # plt.legend()
    # plt.xlabel('alpha')
    # plt.ylabel('Accuracy score')
    # plt.tight_layout()
    # plt.show()
    # plt.figure(figsize=(25, 15))
    # columns = X_train.columns
    # plot_tree(best_post_tree,
    #           feature_names=columns,
    #           class_names=['Low', 'Medium', 'High'],
    #           filled=True,
    #           rounded=True,
    #           fontsize=8,
    #           precision=2
    #           )
    # plt.title('Decision Tree with post pruning', fontdict={'fontsize': 20})
    # plt.tight_layout()
    # plt.show()
    return best_post_tree

post_prune_DT = post_prune_analysis_DT(X_train, X_test, y_clf_train, y_clf_test)
post_prune_metrics = evaluate_tree_performance(post_prune_DT,
                                            X_test, y_clf_test, "Post-pruned Tree")
#%%
def unpruned_decision_tree_analysis(X_train, X_test, y_train, y_test):
    # Unpruned tree performance (baseline)
    dt_unpruned = DecisionTreeClassifier(random_state=5805)
    dt_unpruned.fit(X_train, y_train)
    y_pred_unpruned = dt_unpruned.predict(X_test)
    y_pred_prob_unpruned = dt_unpruned.predict_proba(X_test)[:, 1]
    ################################################################
    # performance
    ########################

    post_accuracy = accuracy_score(y_test, y_pred_unpruned)

    plt.figure(figsize=(25, 15))
    columns = X_train.columns
    plot_tree(dt_unpruned,
              feature_names=columns,
              class_names=['Low', 'Medium', 'High'],
              filled=True,
              rounded=True,
              fontsize=8,
              precision=2
              )
    plt.title('Un-pruned Decision Tree', fontdict={'fontsize': 20})
    plt.tight_layout()
    plt.show()
    print(f'Accuracy on train set: {round(post_accuracy, 2)}')
    return dt_unpruned

unpruned_DT = unpruned_decision_tree_analysis(X_train, X_test, y_clf_train, y_clf_test)
unpruned_metrics = evaluate_tree_performance(unpruned_DT,
                                          X_test, y_clf_test, "Unpruned Tree")

#%%
############Metrics###############
comparison_df = pd.DataFrame({
    'Pre-pruned': pre_prune_metrics,
    'Post-pruned': post_prune_metrics,
    'Unpruned': unpruned_metrics
}).T

print("\nModel Comparison:")
print(comparison_df[['accuracy', 'precision', 'recall', 'f1', 'auc']])

######Plots
def plot_roc_curves_for_all_trees(pre_prune_tree, post_prune_tree, unpruned_tree, X_test, y_test):
    trees = {
        'Pre-pruned': pre_prune_tree,
        'Post-pruned': post_prune_tree,
        'Unpruned': unpruned_tree
    }


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    n_classes = len(np.unique(y_test))
    colors = ['#FF0000', '#00FF00', '#0000FF',
              '#FF00FF', '#00FFFF', '#FFFF00',
              '#FF8000', '#FF0080', '#80FF00',
              '#0080FF', '#8000FF', '#00FF80']


    tree_styles = {
        'Pre-pruned': '-',  # solid
        'Post-pruned': '--',  # dashed
        'Unpruned': '-.'  # dash-dot
    }

    # Plot One-vs-All ROC curves
    for (tree_name, tree_model), color in zip(trees.items(), colors):
        y_pred_prob = tree_model.predict_proba(X_test)
        treestyle = tree_styles[tree_name]
        #One vs All
        ovr_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
        print(f"\nOne-vs-Rest AUC for {tree_name}: {ovr_auc:.4f}")


        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int),
                                    y_pred_prob[:, i])
            class_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr,
                     label=f'{tree_name} - Class {i} (AUC = {class_auc:.2f})',
                     alpha=0.7,
                     color=colors[i % len(colors)],
                     linestyle=treestyle,
                     linewidth=2.5,
                     )

    ax1.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('One-vs-Rest ROC Curves')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)

    colori = 0
    for (tree_name, tree_model), color in zip(trees.items(), colors):
        y_pred_prob = tree_model.predict_proba(X_test)
        treestyle = tree_styles[tree_name]
        # One vs One
        ovo_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovo')
        print(f"One-vs-One AUC for {tree_name}: {ovo_auc:.4f}")

        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                #creating mask so that making class other than class we are taking as 0 and it as 1. Making it basically binary
                mask = np.logical_or(y_test == i, y_test == j)
                if np.any(mask):
                    y_paired = y_test[mask]
                    proba_pair = y_pred_prob[mask]

                    y_binary = (y_paired == i).astype(int)
                    probas = proba_pair[:, i] / (proba_pair[:, i] + proba_pair[:, j])
                    find_nan = ~np.isnan(probas)
                    # print(f"get probas: {np.isnan(probas).sum()} and find nan: {find_nan} and sum {find_nan.sum().sum()}")
                    if find_nan.sum()>0:
                        fpr, tpr, _ = roc_curve(y_binary[find_nan], probas[find_nan])
                        pair_auc = auc(fpr, tpr)
                        ax2.plot(fpr, tpr,
                                 label=f'{tree_name} - Class {i} vs {j} (AUC = {pair_auc:.2f})',
                                 alpha=0.7,
                                 color=colors[colori % len(colors)],
                                 linewidth=2.5,
                                 linestyle=treestyle,
                                 )
                        colori = colori+1

    ax2.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
    ax2.set_xlabel('False Positive Rate(FPR)')
    ax2.set_ylabel('True Positive Rate(TPR)')
    ax2.set_title('One-vs-One ROC Curves')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    results = []
    for tree_name, tree_model in trees.items():
        y_pred_prob = tree_model.predict_proba(X_test)
        ovr_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
        ovo_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovo')
        results.append({
            'Model': tree_name,
            'OvR_AUC': ovr_auc,
            'OvO_AUC': ovo_auc
        })

    return pd.DataFrame(results)
results_auc_dt = plot_roc_curves_for_all_trees(pre_pruning_results_decision_tree,
                             post_prune_DT,
                             unpruned_DT,
                             X_test,
                             y_clf_test)

print(f"AUC Results:")
print(pd.DataFrame(results_auc_dt))
#%%
#########################
# Other classifiers
#########################

def find_optimal_c(X_train, y_train):
    c_values = np.logspace(-4, 4, 20)  # Creates range from 10^-4 to 10^4

    param_grid = {
        'C': c_values,
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    lr_grid = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1
    )

    lr_grid.fit(X_train, y_train)

    # Plot C value vs mean test score for each penalty
    plt.figure(figsize=(10, 6))

    # Get mean test scores for each C value and penalty
    results = pd.DataFrame(lr_grid.cv_results_)

    # Plot for each penalty
    for penalty in ['l1', 'l2']:
        penalty_mask = results['param_penalty'] == penalty
        c_scores = results[penalty_mask].groupby('param_C')['mean_test_score'].mean()
        plt.semilogx(c_scores.index, c_scores.values,
                     marker='o', label=f'penalty={penalty}')

    plt.xlabel('C value (log scale)')
    plt.ylabel('Mean F1 Score')
    plt.title('C Value vs Model Performance')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"Best parameters for Logistic Regression: {lr_grid.best_params_}")
    return lr_grid.best_params_


def find_optimal_k(X_train, y_train):

    # Calculate suggested max k value based on square root of training samples
    n_samples = len(y_train)
    max_k = int(np.sqrt(n_samples))

    # Create range of k values (odd numbers to avoid ties)
    k_values = np.arange(1, max_k + 1, 2)
    k_scores = []
    k_std = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='f1_weighted')
        k_scores.append(scores.mean())
        k_std.append(scores.std())

    # Plot elbow curve with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, k_scores, yerr=k_std, fmt='o-', capsize=5)
    plt.xlabel('k value')
    plt.ylabel('F1 Score')
    plt.title('Elbow Method for KNN')
    plt.grid(True)

    # Find elbow point using maximum curvature
    diff1 = np.diff(k_scores)
    diff2 = np.diff(diff1)
    elbow_idx = np.argmax(np.abs(diff2)) + 1
    optimal_k = k_values[elbow_idx]

    plt.axvline(x=optimal_k, color='r', linestyle='--',
                label=f'Optimal k={optimal_k}')
    plt.legend()
    plt.show()
    print(f"Optimal k for KNN: {optimal_k}")
    return optimal_k


def evaluate_classifier(classifier, X_train, X_test, y_train, y_test, classifier_name, cv=5):
    classifier.fit(X_train, y_train)

    # Predictions
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)

    cm = confusion_matrix(y_test, y_pred)


    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    #######Calculate Metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    specificity = {}
    f1 = f1_score(y_test, y_pred, average='weighted')

    # specificity calculation
    n_classes = len(np.unique(y_test))
    for i in range(n_classes):
        true_negative = np.sum(np.delete(cm, i, 0).sum(0))
        false_positive = np.sum(cm[:, i]) - cm[i, i]
        specificity[i] = true_negative / (true_negative + false_positive)

    # 3. ROC Curve and AUC
    plt.figure(figsize=(8, 6))
    auc_scores = []

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int),
                                y_pred_proba[:, i])
        auc_score = auc(fpr, tpr)
        print(f"AUC score: {auc_score}")
        auc_scores.append(auc_score)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (One-vs-Rest) - {classifier_name}')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    ovo_auc_scores = []

    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            mask = np.logical_or(y_test == i, y_test == j)
            if np.any(mask):
                y_pair = y_test[mask]
                prob_pair = y_pred_proba[mask]

                probas = prob_pair[:, i] / (prob_pair[:, i] + prob_pair[:, j])
                y_binary = (y_pair == i).astype(int)
                find_nan = ~np.isnan(probas)
                # print(f"get probas: {np.isnan(probas).sum()} and find nan: {find_nan} and sum {find_nan.sum().sum()}")
                if find_nan.sum() > 0:
                    fpr, tpr, _ = roc_curve(y_binary[find_nan], probas[find_nan])
                    auc_score = auc(fpr, tpr)
                    print(f"One-vs-One AUC score for class {i} vs {j}: {auc_score}")
                    ovo_auc_scores.append(auc_score)
                    plt.plot(fpr, tpr, label=f'Class {i} vs {j} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (One-vs-One) - {classifier_name}')
    plt.legend()
    plt.show()

    # Stratified K-fold Cross Validation
    cv_scores = cross_val_score(classifier, X_train, y_train,
                                cv=StratifiedKFold(n_splits=cv))

    return {
        'classifier_name': classifier_name,
        'precision': precision,
        'recall': recall,
        'specificity': np.mean(list(specificity.values())),
        'f1_score': f1,
        'auc_scores': np.mean(auc_scores),
        'ovo_auc_scores': np.mean(ovo_auc_scores),
        'cv_scores': cv_scores,
        'confusion_matrix': cm
    }


def train_and_evaluate_all_classifiers(X_train, X_test, y_train, y_test):

    results = []

    # Logistic Regression
    print("\nLogistic Regression")
    print("=" * 50)
 #   best_lr_params = find_optimal_c(X_train, y_train)
    best_lr_params= {'C': 0.012742749857031334, 'penalty': 'l1', 'solver': 'saga'}
    lr = LogisticRegression(random_state=5805, **best_lr_params, max_iter=1000)
    #Best parameters for Logistic Regression: {'C': 0.012742749857031334, 'penalty': 'l1', 'solver': 'saga'}
    results.append(evaluate_classifier(lr, X_train, X_test, y_train, y_test,
                                       "Logistic Regression"))

    #  KNN with Elbow Method
    print("\nKNN")
    print("=" * 50)
#    optimal_k = find_optimal_k(X_train, y_train)
#Optimal k for KNN: 5 - obtained by running the above function and using elbow method
    optimal_k = 5
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    results.append(evaluate_classifier(knn, X_train, X_test, y_train, y_test,
                                      "KNN"))
    # SVM with different kernels
    # print("\nSVM")
    # print("=" * 50)
    # kernels = ['linear', 'poly', 'rbf']
    # for kernel in kernels:
    #     svm = SVC(kernel=kernel, probability=True, random_state=5805)
    #     results.append(evaluate_classifier(svm, X_train, X_test, y_train, y_test,
    #                                        f"SVM-{kernel}"))

    # Naive Bayes
    print("\nNaive Bayes")
    print("=" * 50)
    nb = GaussianNB()
    results.append(evaluate_classifier(nb, X_train, X_test, y_train, y_test,
                                       "Naive Bayes"))

    # Random Forest with Ensemble Methods
    # print("\nRandom Forest and Ensembles")
    # print("=" * 50)
    #
    # # Basic Random Forest
    # rf = RandomForestClassifier(random_state=5805)
    # results.append(evaluate_classifier(rf, X_train, X_test, y_train, y_test,
    #                                    "Random Forest"))
    #
    # # Bagging
    # bagging = BaggingClassifier(random_state=5805)
    # results.append(evaluate_classifier(bagging, X_train, X_test, y_train, y_test,
    #                                    "Bagging"))
    #
    # # Boosting
    # boosting = GradientBoostingClassifier(random_state=5805)
    # results.append(evaluate_classifier(boosting, X_train, X_test, y_train, y_test,
    #                                    "Boosting"))
    #
    # # Stacking
    # estimators = [
    #     ('rf', RandomForestClassifier(random_state=5805)),
    #     ('knn', KNeighborsClassifier()),
    #     ('svm', SVC(probability=True))
    # ]
    # stacking = StackingClassifier(estimators=estimators,
    #                               final_estimator=LogisticRegression())
    # results.append(evaluate_classifier(stacking, X_train, X_test, y_train, y_test,
    #                                    "Stacking"))

    # Neural Network
    print("\nNeural Network")
    print("=" * 50)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000,
                        random_state=42)
    results.append(evaluate_classifier(mlp, X_train, X_test, y_train, y_test,
                                       "Neural Network"))

    # Create comparison table
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('classifier_name')
    comparison_df['cv_mean'] = comparison_df['cv_scores'].apply(np.mean)
    comparison_df['cv_std'] = comparison_df['cv_scores'].apply(np.std)

    print("\nClassifier Comparison:")
    comparison_table = comparison_df[['precision', 'recall', 'specificity',
                                      'f1_score', 'auc_scores', 'cv_mean', 'cv_std']]
    print(comparison_table)

    # Plot comparison
    plt.figure(figsize=(12, 6))
    metrics = ['precision', 'recall', 'f1_score', 'auc_scores']
    comparison_df[metrics].plot(kind='bar')
    plt.title('Classifier Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return comparison_df


# Usage
#
# ###########check if it is a balanced classification target############################
plt.figure()
sns.countplot(data=final_df, x='price_category')
plt.title('Check for dataset Balance')
plt.xlabel('Price Category')
plt.ylabel('Count')
plt.show()
X_train, X_test, y_reg_train,y_reg_test, y_clf_train, y_clf_test = prepare_data(final_df)

results_df = train_and_evaluate_all_classifiers(X_train, X_test, y_clf_train, y_clf_test)


################################

#%%
######################################################################################################################
#PHASE 4 Clustering and Association
######################################################################################################################

#%%
#### K-Means##################
###Since I obtained optimal k value for KNN algorithm
# taking a thumb rule of max_clusters = int(np.sqrt(n_samples/2))
#I will be taking optimal k value obtained(5) +- some range(5) - To make the process faster and efficient

def analyze_kmeans_with_optimal_k(X_train):
    #optimal k from KNN = 5
    optimal_k_knn = 5
    range_around_k = 5
    min_k = max(2, optimal_k_knn - range_around_k)
    max_k = optimal_k_knn+range_around_k
    k_values = range(min_k, max_k + 1)
    silhouette_scores_kmeans = []
    intertia_kmeans = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=5805, n_init=10)
        #took n_init as 10 to get stable results as well as some trade of for computation time
        labels = kmeans.fit_predict(X_train)
        silhouette_avg = silhouette_score(X_train, labels)
        silhouette_scores_kmeans.append(silhouette_avg)
        intertia_kmeans.append(kmeans.inertia_)

        print(f"K={k}: Silhouette Score = {silhouette_avg:.3f}")

    ###plot silhouette score
    plt.figure(figsize=(12,8))
    plt.plot(k_values, silhouette_scores_kmeans, 'bo-', linewidth=2.8)
    plt.axvline(x=optimal_k_knn, color='r', linestyle='--', label='Initial Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score vs Number of Clusters')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    ###plot elbow curve
    plt.figure(figsize=(12,8))
    plt.plot(k_values, intertia_kmeans, 'bo-', linewidth=2.8)
    plt.axvline(x=optimal_k_knn, color='r', linestyle='--', label='Initial Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Intertia Score')
    plt.title('Elbow Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    optimal_k = k_values[np.argmax(silhouette_scores_kmeans)]
    print(f"Optimal K={optimal_k}")
    return optimal_k, silhouette_scores_kmeans, intertia_kmeans

optimal_k, silhouette_scores_kmeans, intertia_kmeans = analyze_kmeans_with_optimal_k(X_train)
print(f"Suggested optimal K from K-means analysis: {optimal_k}")

#%%
#####now fit the clusters with optimal k value
##optimal k = 3
def cluster_analysis_optimal_k(X_train, optimal_k):
    kmeans = KMeans(n_clusters=optimal_k, random_state=5805)
    clusters_lbel = kmeans.fit_predict(X_train)
    cluster_statastics = pd.DataFrame()
    for i in range(optimal_k):
        cluster_data = X_train[clusters_lbel == i]
        cluster_statastics[f'Cluster_{i}'] = cluster_data.mean()

    return clusters_lbel, cluster_statastics

optimal_k = 3 ####obtained from above results
cluster_labels, cluster_statastics = cluster_analysis_optimal_k(X_train, optimal_k)
print("\nCluster Statistics:")
print(cluster_statastics)

#%%
#######Apriori Analysis
#Taking min_support = 0.5 and min_confidence = 0.6
def perform_property_apriori_apartment_rent(X_train, min_support=0.2, min_confidence=0.5):
    bin_features = [
        'AC', 'Cable or Satellite', 'Dishwasher', 'Elevator',
        'Fireplace', 'Gym', 'Parking', 'Patio/Deck', 'Playground',
        'Pool', 'Refrigerator', 'Storage', 'TV', 'Washer Dryer',
        'Wood Floors'
    ]
    property_features = X_train[bin_features].copy()
    property_features['Large_Property'] = (X_train['square_feet'] > X_train['square_feet'].median()).astype(int)
    property_features['Many_Bathrooms'] = (X_train['bathrooms'] > X_train['bathrooms'].median()).astype(int)
    property_features['Many_Bedrooms'] = (X_train['bedrooms'] > X_train['bedrooms'].median()).astype(int)
    frequent_itemsets = apriori(property_features,
                               min_support=min_support,
                               use_colnames=True)

    rules = association_rules(frequent_itemsets,
                              metric="confidence",
                              min_threshold=min_confidence)
    rules = rules.sort_values('lift', ascending=False)
    print(f"\nTotal number of rules found: {len(rules)}")
    print(f"\nTop {len(rules)} Strongest Association Rules:")
    for idx, rule in rules.head(10).iterrows():
        print(f"\nRule {idx + 1}:")
        print(f"IF property has: {', '.join(rule['antecedents'])}")
        print(f"THEN it's likely to have: {', '.join(rule['consequents'])}")
        print(f"Support: {rule['support']:.3f} ({rule['support'] * 100:.1f}% of all properties)")
        print(f"Confidence: {rule['confidence']:.3f} ({rule['confidence'] * 100:.1f}% probability)")
        print(f"Lift: {rule['lift']:.2f}x more likely than random chance")

    return rules, frequent_itemsets

rules, itemsets = perform_property_apriori_apartment_rent(X_train)
