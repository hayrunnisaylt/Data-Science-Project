import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def read_dataset(file_path:str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    
    #Get row and column numbers.
    print('Number of quantity:',len(df))
    print('Number of columns:',len(df.columns))

    #Returns statistical information of columns.
    print('\nStatistical information of columns:')
    print(df.describe())
    return df
file_path = 'electric_vehicle_population_data.csv'
dataset = read_dataset(file_path)

#Calculates correlation matrix.
def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = pd.get_dummies(df)
    correlation_matrix = df_encoded.corr()
return correlation_matrix

veri_seti = pd.read_csv('electric_vehicle_population_data.csv')
korelasyon_matrisi = calculate_correlation_matrix(veri_seti)
print('Kolerasyo Matrisi:')
print(korelasyon_matrisi)

#Calculates the distribution.
def get_distribution(df: pd.DataFrame,column_name: str) ->np.ndarray:
    column_values = df[column_name].values
    return np.histogram(column_values, bins = 12)[0]
column_name = "reading score"
distribution = get_distribution(dataset, column_name)

#Using seaborn plots a one-dimensional histogram of the column I select.
def plt_distribution_seaborn(df: pd.DataFrame, column_name: str):
    plt.figure(figsize=(8,6))
    sns.histplot(df[column_name], kde = False, color = 'skyblue', bins = 12)
    plt.xlabel('Sıklık')
    plt.title('Dagilimi' + column_name)
    plt.show()
column_name = 'reading score'
plt_distribution_seaborn(dataset, column_name)

#Draws the 2D histogram and 2D scatter plot of 2 selected columns using seaborn.
def plt_2d(df: pd.DataFrame, column1: str, column2: str):
    plt.figure(figsize=(12,6))
    #Histogram
    plt.subplot(1,2,1)
    plt.hist2d(df[column1], df[column2], bins = 12, cmap='Blues')
    plt.colorbar(label = 'Dağilimi')
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title('Histogram')
    #Scatter Plot
    plt.subplot(1,2,2)
    plt.scatter(df[column1], df [column2]), alpha =
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title('Scatter Plot')
    plt.tight_layout()
    plt.show()
plt_2d(dataset, 'reading score', 'writing score')

#It visualizes the correlation matrix we obtained as a heatmap plot.
def heatmap_corelation_matrix(correlation_matrix: pd.DataFrame):
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = ".2f")
    plt.title('Kolerasyon Matrisi')
    plt.show()
file_path = 'exams.csv'
dataset = read_dataset(file_path)
correlation_matrix = calculate_correlation_matrix(dataset)
heatmap_corelation_matrix(correlation_matrix)
