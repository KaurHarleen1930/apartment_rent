#%%
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.stats import spearmanr
from seaborn import color_palette
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import warnings

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.options.display.float_format = "{:,.2f}".format
warnings.filterwarnings("ignore")


#%%
sns.set(
    style='whitegrid',
    palette='bright',
    font='serif',
    rc={
        'figure.figsize': (15, 12),
        'axes.titlesize': 15,
        'axes.titlecolor': 'blue',
        'axes.labelsize': 12,
        'axes.labelcolor': 'darkred',
        'axes.titlepad': 10,
        'font.family': 'serif',
        'grid.color': 'gray',
        'grid.alpha': 0.3
    }
)
url = 'https://github.com/KaurHarleen1930/apartment_rent-private-repo/raw/refs/heads/main/apartments_for_rent_classified_100K.csv'
#######please uncomment line 42 and comment line 40 if you want to read data locally
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
print(df.head())
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
    sns.histplot(apts[col], kde=True, bins=30, color='blue')
    plt.title(f"Initial Histogram plot of {col}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

    #box plot
    sns.boxplot(x=apts[col], color='orange')
    plt.title(f"Boxplot of {col} to check for outliers")
    plt.xlabel(col)
    plt.show()

    #violin plot
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
####reg plot - Shows relationship between rank and value

plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.regplot(
        data=apts,
        x=apts[feature].rank(),
        y=feature,
        scatter_kws={'alpha': 0.5, 'color': sns.color_palette("husl", 8)[i-1]},
        line_kws={'color': 'purple', 'linewidth': 1.5},
        ci=95
    )
    plt.title(f'Regression Plot of {feature}')
    plt.xlabel('Rank')
    plt.ylabel(feature)
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
###dist plot with rug
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(
        data=apts,
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
####Shows cumulative probability
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.ecdfplot(
        data=apts,
        x=feature,
        linewidth=2,
        color=sns.color_palette("bright", 8)[i-1]
    )
    plt.title(f'ECDF of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Cumulative Proportion')
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

    plt.figure(figsize=(12, 6))
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
#line plot

plt.plot(apts['square_feet'], apts['price'], marker='o')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Price vs Square Feet')
plt.show()
###

sns.kdeplot(data=apts, x='square_feet', y='price', fill=True)
plt.title('2D Distribution: Price vs Square Feet')
plt.show()
# Count Plot

apts['price_category'] = pd.qcut(apts['price'], 3, labels=['Low', 'Medium', 'High'])
sns.countplot(data=apts, x='bedrooms', hue='price_category')
plt.title('Count of Properties by Bedrooms and Price Category')

plt.tight_layout()
plt.show()

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

#%%hexbin plot
fig = plt.figure(figsize=(20, 20))
plot_num = 1

for i in range(len(numerical_cols)):
    for j in range(i+1, len(numerical_cols)):
        plt.subplot(3, 2, plot_num)
        plt.hexbin(apts[numerical_cols[i]], apts[numerical_cols[j]],
                  gridsize=10, cmap='cividis')
        plt.colorbar(label='Count')
        plt.xlabel(numerical_cols[i])
        plt.ylabel(numerical_cols[j])
        plt.title(f'Hexbin: {numerical_cols[i]} vs {numerical_cols[j]}')
        plot_num += 1

plt.tight_layout()
plt.show()

#%% density contour plot
# fig = plt.figure(figsize=(20, 20))
# plot_num = 1
#
# for i in range(len(numerical_cols)):
#     for j in range(i+1, len(numerical_cols)):
#         plt.subplot(3, 2, plot_num)
#         sns.kdeplot(data=apts, x=numerical_cols[i], y=numerical_cols[j],
#                    levels=20, cmap='viridis', fill=True)
#         plt.xlabel(numerical_cols[i])
#         plt.ylabel(numerical_cols[j])
#         plt.title(f'Density Contour: {numerical_cols[i]} vs {numerical_cols[j]}')
#         plot_num += 1
#
# plt.tight_layout()
# plt.show()

#%% scatter plot with regression line
fig = plt.figure(figsize=(20, 20))
plot_num = 1

for i in range(len(numerical_cols)):
    for j in range(i+1, len(numerical_cols)):
        plt.subplot(3, 2, plot_num)
        sns.regplot(data=apts, x=numerical_cols[i], y=numerical_cols[j],
                   scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
        plt.xlabel(numerical_cols[i])
        plt.ylabel(numerical_cols[j])
        plt.title(f'Scatter with Regression: {numerical_cols[i]} vs {numerical_cols[j]}')
        plot_num += 1

plt.tight_layout()
plt.show()

#%%
fig = plt.figure(figsize=(20, 25))
plot_num = 1

for feature in ['price', 'square_feet']:

    plt.subplot(4, 2, plot_num)
    sns.stripplot(data=apts[:1000], x='bedrooms', y=feature, jitter=True, size=10)
    plt.title(f'Strip Plot: {feature} by Bedrooms')

    plt.subplot(4, 2, plot_num + 1)
    sns.stripplot(data=apts[:1000], x='bathrooms', y=feature, jitter=True, size=10)
    plt.title(f'Strip Plot: {feature} by Bathrooms')


    plt.subplot(4, 2, plot_num + 2)
    sns.swarmplot(data=apts[:1000], x='bedrooms', y=feature, size=10)
    plt.title(f'Swarm Plot: {feature} by Bedrooms')

    plt.subplot(4, 2, plot_num + 3)
    sns.swarmplot(data=apts[:1000], x='bathrooms', y=feature, size=10)
    plt.title(f'Swarm Plot: {feature} by Bathrooms')

    plot_num += 4

plt.tight_layout()
plt.show()


# plt.figure(figsize=(10, 10))
correlation = apts[numerical_cols].corr()
sns.clustermap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Clustered Correlation Matrix')
plt.tight_layout()
plt.show()

#%% categorical feature analysis

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

colors = sns.color_palette('husl', n_colors=10)


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
apts.drop(columns = 'cluster',axis=1, inplace=True)
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

for col in numerical_cols:
    # #histogram plot
    # plt.figure(figsize=(8, 4))
    # sns.histplot(apts[col], kde=True, bins=30, color='blue')
    # plt.title(f"Histogram plot of {col} after removing outliers")
    # plt.xlabel(col)
    # plt.ylabel('Frequency')
    # plt.show()

    #box plot
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=apts[col], color='orange')
    plt.title(f"Boxplot of {col} after removing outliers")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

    # #violin plot after removing outliers
    # plt.figure(figsize=(8, 4))
    # violin = sns.violinplot(
    #     data=apts,
    #     y=col,
    #     width=0.8,
    #     inner='box',
    #     linewidth=2,
    #     color='skyblue'
    # )
    # plt.title(f'Violin plot: Removed outliers for {col}', pad=20)
    # plt.ylabel(col)
    # plt.xticks(rotation=0)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()

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

#kdeplot - latitude and longitude
plt.figure(figsize=(10, 6))
sns.kdeplot(x='longitude', y='latitude', data=apts, shade=True, cmap='viridis', alpha=0.8)
plt.title("Geospatial Density of Apartments")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
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
            palette=sns.color_palette('coolwarm', n_colors=8))
ax1.set_ylabel('Average Price (Bars)')

ax2 = ax1.twinx()

sns.scatterplot(x = 'state', y = 'count', data = state_counts,  ax=ax2)
ax2.set_ylabel('Number of Datapoints (Points)')

plt.suptitle("State, Price and Count Analysis")
plt.show()

### Analysis based on number of bedrooms and bathrooms
sns.set(font_scale=1,
palette = 'Set2',
style = 'whitegrid',
)
bathrooms = apts.groupby('state', as_index=False)['bathrooms'].mean().sort_values('bathrooms')
bedrooms = apts.groupby('state', as_index=False)['bedrooms'].mean().sort_values('bedrooms')

bedrooms_with_price = bedrooms.merge(state_means, how='inner', on='state')
bedrooms_bathrooms_price = bedrooms_with_price.merge(bathrooms, how='inner', on='state')

fig, ax1 = plt.subplots(figsize=(16,6))


ax2 = ax1.twinx()
sns.barplot(x = 'state', y='bedrooms', data=bedrooms_bathrooms_price, ax=ax1,
            palette=sns.color_palette('coolwarm', n_colors=5))
ax1.set_ylabel('Average Bathrooms [marker], Average Bedrooms(Bars)')
sns.scatterplot(x = 'state', y = 'bathrooms',data = bedrooms_bathrooms_price,  ax=ax1,
                palette=sns.color_palette('coolwarm', n_colors=5))

sns.lineplot(x = 'state', y = 'price',data = bedrooms_bathrooms_price, ax=ax2,
             palette=sns.color_palette('coolwarm', n_colors=5))
ax2.set_ylabel('Average Price (Line)')
plt.suptitle("State, Bedroom, Bathroom vs Price Analysis")
plt.tight_layout()
plt.show()

##square feet analysis
sqft_apts = apts.groupby('state', as_index=False)['square_feet'].mean()

sqft_apts = sqft_apts.merge(state_means, how='inner', on='state')

sqft_apts['dollar_per_sqft'] = sqft_apts['price']/sqft_apts['square_feet']
sqft_apts = sqft_apts.sort_values('dollar_per_sqft')

fig, ax1 = plt.subplots(figsize=(16,6))
sns.barplot(x = 'state', y='dollar_per_sqft', data=sqft_apts,
            palette=sns.color_palette('coolwarm', n_colors=5))
ax1.set_ylabel('Dollars Per Square Foot')

ax2 = ax1.twinx()
sns.scatterplot(x = 'state', y = 'square_feet',data = sqft_apts,  ax=ax2,
                palette=sns.color_palette('coolwarm', n_colors=5))
ax2.set_ylabel('Average Square Feet')
plt.suptitle('Dollars Per Square Foot vs State Price analysis')
plt.show()

#%%State vs Price Line plot Analysis
sns.lineplot(x = 'state', y = 'price',data = state_means,
            palette=sns.color_palette('bright', n_colors=8),
             lw = 3)
plt.title("State vs Price Analysis")
plt.tight_layout()
plt.show()

#%%State vs Square Feet plot Analysis
sns.lineplot(x = 'state', y = 'square_feet',data = sqft_apts,
            palette=sns.color_palette('bright', n_colors=8),
             lw = 3)
plt.title("State vs Square Feet Analysis")
plt.tight_layout()
plt.show()


#%%
##########################
##Correlation Analysis
##########################
correlation = apts[numerical_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(apts[numerical_cols], diag_kind='kde', plot_kws={'alpha': 0.7},
             palette=sns.color_palette('coolwarm', n_colors=5))
plt.show()

#%%%%
##################
#Join plot with KDE
###################
fig = plt.figure(figsize=(15,6))
g = sns.jointplot(
    data=apts,
    x='price',
    y='square_feet',
    kind='scatter',
    marginal_kws={'color': 'blue'},
    joint_kws={'alpha': 0.7},
    height=10
)
g.plot_joint(sns.kdeplot, cmap='viridis', levels=5)

plt.suptitle(f'Joint Plot price vs square_feet',
             y=1.02, fontsize=12)
plt.tight_layout()
plt.show()

g = sns.PairGrid(apts[numerical_cols])
g.map_diag(sns.histplot)
g.map_upper(sns.scatterplot, alpha=0.5)
g.map_lower(sns.kdeplot)
plt.suptitle('Pair Grid with Histogram, Scatter, and KDE Plots', y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

#%% created tables
categorical_cols = ['has_photo', 'pets_allowed', 'state', 'fee', 'amenities']
for column in categorical_cols:
    if column=='amenities':
        # Split the amenities column and count each unique amenity
        amenities_list = apts['amenities'].str.split(',').sum()
        amenities_count = Counter(amenities_list).most_common(15)
        dfa  = pd.DataFrame(amenities_count)
        x = dfa[0]
        y=dfa[1]
    elif column == 'state':
        dfs = pd.DataFrame(apts['state'].value_counts().sort_values(ascending=False).head(15).reset_index())
        x = dfs['state']
        y = dfs['count']
    else:
        value_counts = apts[column].value_counts()
        x = value_counts.index
        y=value_counts.values

    freq_df = pd.DataFrame({
        'Category': x,
        'Count': y,
        'Percentage': (y / len(apts) * 100).round(2)
    })
    print(f"Frequency Table for {column}")
    print(freq_df)
    table = PrettyTable()
    table.field_names = freq_df.columns
    for row in freq_df.itertuples(index=False):
        table.add_row(row)
    print(table)

#######################
##PCA Analysis
######################
#%%
##########################
#Encoding before PCA
############################
def convert_categorical_data(apts, col, remove):
    apts[col] = apts[col].str.strip()
    apts = pd.concat([apts, apts[col].str.get_dummies(sep=',')], axis=1)
    if remove == True:
        apts = apts.drop([col], axis=1)
    return apts


apts = convert_categorical_data(apts, 'amenities', True)

apts['fee'] = apts['fee'].map({'Yes':1, 'No':0})
apts['has_photo'] = apts['has_photo'].map({'Yes':2, 'Thumbnail':1,'No':0})

apts['pets_allowed'] = apts['pets_allowed'].str.strip()
dummies_pets = apts['pets_allowed'].str.get_dummies(sep=',')
dummies_pets = pd.DataFrame(dummies_pets)
dummies_pets.rename(columns = {'Cats':'Pet_Cats','Dogs':'Pet_Dogs','None':'Pet_None'}, inplace = True)
apts = pd.concat([apts, dummies_pets], axis=1)
apts.drop(inplace = True, axis = 1, columns='pets_allowed')
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


apts = encode_cityname(apts)
apts.drop(columns='cityname', inplace=True)


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

apts = encode_state(apts)
apts.drop(columns='state', inplace=True)
apts.drop(columns = 'source', inplace=True)
apts.drop(columns = 'time', inplace=True)

#%%
def prepare_data(df, target = 'price'):
    scaler = StandardScaler()
    features_to_scale = [
        'bathrooms',
        'bedrooms',
        'square_feet',
        'cityname_mean_price',
        'cityname_price_rank',
        'state_mean_price',
    ]
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    X = df.drop(columns=[target], axis=1)
    y_reg = df[target]


    #split data
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=5764
    )
    return X_train, X_test, y_reg_train, y_reg_test
# PCA Analysis
X_train, X_test, y_reg_train, y_reg_test = prepare_data(apts)

#%%
def dimensionality_reduction(X, y):
    print("PCA Analysis")
    print("=" * 50)

    numeric_X = X.select_dtypes(include=['int64', 'float64'])
    X_scaled = StandardScaler().fit_transform(numeric_X)

    pca = PCA()
    pca.fit_transform(X_scaled)

    explained_variance = pd.DataFrame({
        'Component': range(1, len(pca.explained_variance_ratio_) + 1),
        'Explained_Variance': pca.explained_variance_ratio_,
        'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
    })

    cumm_variance = np.cumsum(pca.explained_variance_ratio_)
    features_95_threshold = np.argmax(cumm_variance >= 0.95) + 1

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(explained_variance['Component'],
             explained_variance['Explained_Variance'],
             'bo-',
             linewidth=2,
             markersize=6)
    plt.title('Explained Variance Ratio Plot',
              fontsize=12,
              fontweight='bold',
              color='blue',
              fontfamily='serif')
    plt.xlabel('Principal Component', fontfamily='serif', color='darkred')
    plt.ylabel('Explained Variance Ratio', fontfamily='serif', color='darkred')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(explained_variance['Component'],
             explained_variance['Cumulative_Variance'],
             'ro-',
             linewidth=2,
             markersize=6)
    plt.axhline(y=0.95, color='k', linestyle='--', label='95% Threshold')
    plt.axvline(x=features_95_threshold, color='k', linestyle='--')
    plt.title('Cumulative Explained Variance',
              fontsize=12,
              fontweight='bold',
              color='blue',
              fontfamily='serif')
    plt.xlabel('Number of Components', fontfamily='serif', color='darkred')
    plt.ylabel('Cumulative Explained Variance', fontfamily='serif', color='darkred')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    try:
        condition_number = np.linalg.cond(X_scaled)
        print(f"\nCondition Number Analysis")
        print("=" * 50)
        print(f"Condition Number: {condition_number:.2f}")
        print(f"\nInterpretation:")
        if condition_number < 100:
            print("- Good conditioning (< 100)")
        elif condition_number < 1000:
            print("- Moderate conditioning (100 - 1000)")
        else:
            print("- Poor conditioning (> 1000)")
    except Exception as e:
        print("\nWarning: Could not compute condition number")
        print(f"Error: {str(e)}")
        condition_number = None

    print("\nPCA Analysis Summary:")
    print("=" * 50)
    print(f"Original number of features: {numeric_X.shape[1]}")
    print(f"Components needed for 95% variance: {features_95_threshold}")
    print(f"Variance explained by first component: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"Total variance explained by all components: {sum(pca.explained_variance_ratio_):.3f}")

    return explained_variance, condition_number, features_95_threshold

X_train = pd.get_dummies(X_train, drop_first=True)
X_train = X_train.dropna()
explained_variance, condition_number, n_components_95 = dimensionality_reduction(X_train, y_reg_train)