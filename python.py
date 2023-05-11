"""
Pandas --> used for data analysis and manupulations
Numpy --> used for numerical operations
Scipy --> used for algortihm optimizations
Matplotlib --> for dara visualization
sklearn.cluster --> For importing KMeans clustering algorithm
sklearn.datasets --> for creating blob sample dataset
"""
import itertools as iter
import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


## The following funciton will help in analyzing data for different countries mentioned in the dataset
def read_dataset(file):
    
    ## Read the data file and then return actual and tranformed dataframes accordingly
    
    # Reading dataset file
    dataset = pd.read_csv(file)

    # Extract countries list
    countries = list(dataset['Country Name'])

    # Now transpose the dataframe and change column names
    data_transpose = dataset.transpose()
    data_transpose.columns = countries

    # Eliminate unwanted data/rows from dataframe
    data_transpose = data_transpose.iloc[4:]
    data_transpose = data_transpose.iloc[:-1]

    # Reset the index column
    data_transpose = data_transpose.reset_index()

    # Rename index column
    data_transpose = data_transpose.rename(columns={"index": "Year"})

    # Convert column 'Year' to data type integer (int)
    data_transpose['Year'] = data_transpose['Year'].astype(int)

    return dataset, data_transpose


data_1, data_2 = read_dataset('population_data.csv')

## Read dataset file and accordingly return dataframes both actual and transformed one.
def read_data(datafile):
    
    # Reading the file
    dtf = pd.read_csv(datafile)
    country_dtf = list(dtf['Country Name'])
    
    # Transforming the data
    transformed_data = dtf.transpose()
    transformed_data.columns = country_dtf
    
    # Eliminate irrelevant data/rows from dataframe
    transformed_data = transformed_data.iloc[4:]
    transformed_data = transformed_data.iloc[:-1]
    
    # Performing the same operations as above i.e., reset index column and then convert column 'Year' to integer data type:

    transformed_data = transformed_data.reset_index()
    transformed_data = transformed_data.rename(columns={"index": "Year"})
    transformed_data['Year'] = transformed_data['Year'].astype(int)

    return dtf, transformed_data


dtf, transformed_data = read_data("population_data_2.csv")

country_name = list(dtf['Country Name'])


"""
The following funciton is calculating mean for our selected indicator, "Population in Urban agglomeration" 
and its mean in a particular year i.e., 2016 as well
"""
def calc_urbanpopulation(population_data_2):

    # Calculate the mean value for indicator "Population in Urban agglomeration":
    calc_urb_population_mean = population_data_2.mean()
    print("Population in Urban Agglomeration Mean: ", calc_urb_population_mean)

    # Calculate the mean value for indicator "Population in Urban agglomeration" in 2016:
    urb_population_mean = population_data_2[population_data_2['Year'] == 2016]
    print("Mean of Population in Urban Agglomeration in 2016 : ", urb_population_mean)

# Calling the function
calc_urbanpopulation(transformed_data)


## Creating 3 clusters
cluster_count = 3


## The following function creates data sample with defined samples and number of cluster counts
def data_sampling(data_sample: int, cluster: int):

    X, y = make_blobs(n_samples=data_sample, centers=cluster, random_state=0)
    return X, y

# Calling the function and assigning X and Y
X, y = data_sampling(200, cluster_count)


## The following function will apply k-means clustering model to the dataset
def kmeans(dataset, cluster):
    kmeans_cluster = KMeans(n_clusters=cluster, random_state=0).fit(dataset)

    return kmeans_cluster

# Calling the function
kmean = kmeans(X, cluster_count)

## the following function will create a visualization for kmeans clustering with their respective labels and defined title
def create_plot(dataset, kmodel, xlabel, ylabel, title):
    plt.scatter(dataset[:, 0], dataset[:, 1], c=kmodel.labels_)
    ## creating a scattered plot
    plt.scatter(kmodel.cluster_centers_[:, 0], kmodel.cluster_centers_[:, 1], marker='D', color='blue')

    ## define labels fonts
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(['Cluster', 'Cluster Center'])
    plt.show()

# Calling the function
create_plot(X, kmean, 'X-axis', 'Y-axis', 'KMean Clustering Algorithm Plot')



# This function creates the visual for specifically defined countries only
def urbanpopulation_shift(population_data_2, countries):

    population_data_2.plot(x = 'Year', y = countries)
    plt.title("% of Total Population in Urban Agglomeration Countries", fontsize=15)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("% of Total Population", fontsize=12)
    plt.show()

# Calling the function
urbanpopulation_shift(data_2, ['Germany', 'France', 'China', 'India', 'Canada'])


## The following function creates another visual for country over time
def create_urbanpopulation_plot(df, country_dtf):
    
    df.plot("Year", country_dtf)
    plt.title(f"{country_dtf}'s Population in Urban Agglomeration", fontsize=15)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("% of Urban Population", fontsize=12)
    plt.legend(["Population in Urban Agglomeration"])
    plt.show()

# Calling the function
create_urbanpopulation_plot(transformed_data, country_name[0])



