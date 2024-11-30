#%%

import pandas as pd
from geopy.exc import GeocoderTimedOut
from geopy.extra.rate_limiter import RateLimiter
from seaborn import color_palette
from sklearn.cluster import KMeans
from geopy.geocoders import Nominatim

import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import time
import warnings

pd.options.display.float_format = "{:,.2f}".format
warnings.filterwarnings("ignore", "use_inf_as_na")
#%%
url = 'https://media.githubusercontent.com/media/KaurHarleen1930/apartment_rent/refs/heads/feature/information_visualization/apartments_for_rent_classified_100K.csv'

df = pd.read_csv(url, sep=";",  encoding='cp1252')
df.drop_duplicates(inplace=True)
df = df[[
    'id', 'amenities', 'bathrooms', 'bedrooms', 'fee', 'has_photo',
    'pets_allowed', 'price', 'square_feet', 'cityname', 'state',
    'latitude', 'longitude', 'source', 'time'
]]

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
# apts = pd.read_csv(url, sep=";",  encoding='cp1252' ,
#                    usecols=['id','amenities', 'bathrooms' , 'bedrooms', 'fee' , 'has_photo', 'pets_allowed' , 'price' , 'square_feet', 'cityname' , 'state', 'latitude', 'longitude' , 'source' , 'time'] ,
#                   dtype={'amenities':'string', 'bathrooms':'Float32' , 'bedrooms':'Float32',
#                         'fee':'string', 'has_photo':'string' ,'pets_allowed':'string', 'price': 'Float64',
#                         'square_feet':'Int64' , 'latitude':'Float64', 'longitude':'Float64' , 'source':'string', 'time':'Int64'} , index_col = 0)
# small_apts = pd.read_csv(url, sep=";",  encoding='cp1252' ,
#                    usecols=['id','amenities', 'bathrooms' , 'bedrooms', 'fee' , 'has_photo', 'pets_allowed' , 'price' , 'square_feet', 'cityname' , 'state', 'latitude', 'longitude' , 'source' , 'time'] ,
#                   dtype={'amenities':'string', 'bathrooms':'Float32' , 'bedrooms':'Float32',
#                         'fee':'string', 'has_photo':'string' ,'pets_allowed':'string', 'price': 'Float64',
#                         'square_feet':'Int64' , 'latitude':'Float64', 'longitude':'Float64' , 'source':'string', 'time':'string'} , index_col = 0)

features = []
y = apts.price

print(apts.head())
print(apts.shape)
print(apts.isnull().sum())
print(apts.duplicated().sum())
pd.set_option('display.max_columns', None)
#%%
print(apts.info())
# null values handle
apts['amenities'] = apts["amenities"].fillna("no amenities available")
apts["pets_allowed"] = apts["pets_allowed"].fillna("None")

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

#%% checking for each numerical column using histogram plot
numerical_cols = ['bathrooms', 'bedrooms', 'price', 'square_feet']

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


#%% reduce extreme values for price and check for all columns
#multivariate outlier
#using IQR outlier removal on price
numerical_cols = ['bathrooms', 'bedrooms', 'price', 'square_feet']
Q1 = apts[numerical_cols].quantile(0.25)
Q3 = apts[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
lower_quantile_range = Q1-1.5*IQR
upper_quantile_range = Q3+1.5*IQR

apts = apts[~((apts[numerical_cols] < lower_quantile_range) | (apts[numerical_cols] > upper_quantile_range)).any(axis=1)]


for col in numerical_cols:
    #histogram plot
    plt.figure(figsize=(8, 4))
    sns.histplot(apts[col], kde=True, bins=30, color='blue')
    plt.title(f"Histogram plot of {col} after removing outliers")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

    #box plot
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=apts[col], color='orange')
    plt.title(f"Boxplot of {col} after removing outliers")
    plt.xlabel(col)
    plt.show()

##Based on our Box-plot there are a lot of outliers in our data,
# so we should address that next. First, lets take a look at the distribution of our data --
# it looks fairly normal, but has a very large tail end and very narrow center band.
# This could be drastically improved by reducing the number of extreme values in our data.
#There are 474 values below 5000
#%% Compare the results before outlier removal and after outlier removal

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

###############
#####Normality test Pending
##################

#%% establishing relationship between different features
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

#%%
plt.figure(figsize=(10, 6))
sns.kdeplot(x='longitude', y='latitude', data=apts, shade=True, cmap='viridis', alpha=0.8)
plt.title("Geospatial Density of Apartments")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
## showing densily populated areas or less densily populated
#%%shows that my data is majorly covering east coast, north and south of USA region. And we are not much covering
#west coast
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
#%%

state_apts = apts.groupby('state')

state_means = state_apts['price'].mean()
state_means = state_means.reset_index()
state_means = state_means.sort_values(['price'], ascending=False)

state_counts = apts.state.value_counts()
state_counts = state_counts.reset_index()

print(state_means.reset_index())
print(state_counts)

#%% Analysis based on state
#take for example DC we can say that we have around 100 apartment listings available and avg price is so high
#which shows supply and demand
fig, ax1 = plt.subplots(figsize=(15,8))

sns.barplot(x = 'state', y = 'price',data = state_means, ax=ax1,
            palette=sns.color_palette('coolwarm', n_colors=8))
ax1.set_ylabel('Average Price (Bars)')

ax2 = ax1.twinx()

sns.scatterplot(x = 'state', y = 'count', data = state_counts,  ax=ax2)
ax2.set_ylabel('Number of Datapoints (Points)')

plt.show()

#An interesting interpretation of this is that we can see a correlation between the number of apartments in the dataset
# (to be interpreted as the number of apartments available) and the price of those apartments.
# This follows supply and demand principles, the less apartments available, the higher that the rents charged will be.
#%% Analysis based on number of bedrooms and bathrooms
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

#create BAR plot
sns.barplot(x = 'state', y='bedrooms', data=bedrooms_bathrooms_price, ax=ax1,
            palette=sns.color_palette('coolwarm', n_colors=5))
ax1.set_ylabel('Average Bathrooms (Points), Average Bedrooms(Bars)')

#create SCATTER plot
sns.scatterplot(x = 'state', y = 'bathrooms',data = bedrooms_bathrooms_price,  ax=ax1,
                palette=sns.color_palette('coolwarm', n_colors=5))

sns.lineplot(x = 'state', y = 'price',data = bedrooms_bathrooms_price, ax=ax2,
             palette=sns.color_palette('coolwarm', n_colors=5))
ax2.set_ylabel('Average Price (Line)')
plt.tight_layout()
#show the plot
plt.show()

#%%
bedrooms_bathrooms_price['bedbathratio'] = bedrooms_bathrooms_price['bedrooms']/bedrooms_bathrooms_price['bathrooms']
bedrooms_bathrooms_price = bedrooms_bathrooms_price.sort_values('price')

fig, ax1 = plt.subplots(figsize=(16,6))

ax2 = ax1.twinx()

sns.barplot(x = 'state', y='bedbathratio', data=bedrooms_bathrooms_price, ax=ax1,
            palette=sns.color_palette('coolwarm', n_colors=5))
ax1.set_ylabel('Bedroom:Bathroom Ratio (Average)')

sns.scatterplot(x = 'state', y = 'price',data = bedrooms_bathrooms_price,  ax=ax2,
                palette=sns.color_palette('coolwarm', n_colors=5))
ax2.set_ylabel('Price')
plt.tight_layout()
plt.show()

#%% square feet

sqft_apts = apts.groupby('state', as_index=False)['square_feet'].mean()

sqft_apts = sqft_apts.merge(state_means, how='inner', on='state')

sqft_apts['dollar_per_sqft'] = sqft_apts['price']/sqft_apts['square_feet']
sqft_apts = sqft_apts.sort_values('dollar_per_sqft')

fig, ax1 = plt.subplots(figsize=(16,6))
sns.barplot(x = 'state', y='dollar_per_sqft', data=sqft_apts,
            palette=sns.color_palette('coolwarm', n_colors=5))
ax1.set_ylabel('Dollars Per Square Foot')

ax2 = ax1.twinx()

#create SCATTER plot
sns.scatterplot(x = 'state', y = 'square_feet',data = sqft_apts,  ax=ax2,
                palette=sns.color_palette('coolwarm', n_colors=5))
ax2.set_ylabel('Average Square Feet')

plt.show()

#%%
# text = apts['amenities'].value_counts()
# print(text.value_counts())
#
# wordcloud_data = text.value_counts().to_dict()
# wordcloud = WordCloud().generate_from_frequencies(wordcloud_data)
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()

all_amenities = apts['amenities'].value_counts()
print(all_amenities)

#%% correlation between different numerical features
correlation = apts[numerical_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(apts[numerical_cols], diag_kind='kde', plot_kws={'alpha': 0.7},
             palette=sns.color_palette('coolwarm', n_colors=5))
plt.show()

#%%count plots for categorical data
categorical_cols = ['has_photo', 'pets_allowed', 'state', 'fee']

for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=apts, palette='bright', order=apts[col].value_counts().index)
    plt.title(f"Count of {col}")
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

#%% amenities
from collections import Counter

# Split the amenities column and count each unique amenity
amenities_list = apts['amenities'].str.split(',').sum()
amenities_count = Counter(amenities_list).most_common(15)  # Top 15 amenities

#%% Create a bar plot
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
#%% changing categorical data using one hot encoding
def convert_categorical_data(apts, col, remove):
    apts[col] = apts[col].str.strip()
    apts = pd.concat([apts, apts[col].str.get_dummies(sep=',')], axis=1)
    if remove == True:
        apts = apts.drop([col], axis=1)
    return apts


apts = convert_categorical_data(apts, 'amenities', True)
print(apts.columns.tolist())

#%% use longitude and latitude to generate neighbourhood or address


#%%
# apts['address_details'] = apts.apply(lambda x: get_address(x['latitude'], x['longitude']), axis=1)
# #%%
# apts['address'] = apts['address_details'].apply(lambda x: x.get('display_name') if x else None)
# apts['neighborhood'] = apts['address_details'].apply(lambda x: x.get('address', {}).get('neighbourhood') if x else None)
# apts['county'] = apts['address_details'].apply(lambda x: x.get('address', {}).get('county') if x else None)
# apts['place_importance']=apts['address_details'].apply(lambda x: x.get('importance') if x else None)
# apts['place_rank']=apts['address_details'].apply(lambda x: x.get('place_rank') if x else None)

# batch_size = 1000  # Adjust batch size to suit your machine's capacity
# processed_batches = []
#
#
# def get_address_details(lat, lon):
#     try:
#         location = geocode((lat, lon))
#         if location:
#             address = location.raw.get('address', {})
#             return {
#                 'address': location.address,
#                 'neighborhood': address.get('neighbourhood'),
#                 'county': address.get('county'),
#                 'place_importance': address.get('importance'),
#                 'place_rank': address.get('place_rank'),
#             }
#     except Exception as e:
#         return None
#
#
# def process_batch(batch):
#     results = []
#     for _, row in batch.iterrows():
#         details = get_address_details(row['latitude'], row['longitude'])
#         if details:
#             results.append(details)
#         else:
#             results.append({'address': None, 'neighborhood': None, 'county': None, 'postcode': None, 'state': None,
#                             'country': None})
#     return pd.DataFrame(results)
#
#
# for i in range(0, len(apts), batch_size):
#     print(f"Processing batch {i // batch_size + 1}...")
#     batch = apts.iloc[i:i + batch_size]
#     processed_batch = process_batch(batch)
#     processed_batches.append(processed_batch)
#
# processed_apts = pd.concat(processed_batches, axis=0).reset_index(drop=True)
#
# apts = pd.concat([apts.reset_index(drop=True), processed_apts], axis=1)
geolocator = Nominatim(user_agent="apartment_rent_analysis")
def geocode_location(lat, lon, retries=3, delay=2):
    for attempt in range(retries):
        try:
            location = geolocator.reverse((lat, lon), language="en")
            return location.raw if location else None
        except GeocoderTimedOut:
            print(f"Timeout. Retrying... ({attempt + 1}/{retries})")
            time.sleep(delay)
        except Exception as e:
            print(f"Error: {e}")
            break
    return None


def process_chunk(chunk):
    results = []
    for _, row in chunk.iterrows():
        lat, lon = row['latitude'], row['longitude']
        result = geocode_location(lat, lon)
        results.append(result)
    return results


# Process large datasets in chunks
def process_large_dataset_in_chunks(df, chunk_size=500, max_workers=10):
    start_time = time.time()
    processed_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            futures.append(executor.submit(process_chunk, chunk))

        for future in futures:
            processed_results.extend(future.result())

    print(f"Processing completed in {time.time() - start_time:.2f} seconds.")
    return processed_results


df_lat_long = pd.DataFrame(apts[['latitude','longitude']])
df_lat_long['address_details'] = process_large_dataset_in_chunks(df_lat_long, chunk_size=500, max_workers=10)

df_lat_long['address'] = df_lat_long['address_details'].apply(lambda x: x['display_name'] if x else None)
df_lat_long['neighborhood'] = df_lat_long['address_details'].apply(lambda x: x.get('address', {}).get('neighbourhood') if x else None)
df_lat_long['county'] = df_lat_long['address_details'].apply(lambda x: x.get('address', {}).get('county') if x else None)
df_lat_long['place_importance']=df_lat_long['address_details'].apply(lambda x: x.get('importance') if x else None)
df_lat_long['place_rank']=df_lat_long['address_details'].apply(lambda x: x.get('place_rank') if x else None)





print(apts['neighborhood','county','address','address_details'])


