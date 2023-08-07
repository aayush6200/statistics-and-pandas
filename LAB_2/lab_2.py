import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


r_path = './dataset/auto.csv'

df = pd.read_csv(r_path)  # creating dataset using pandas to read the file

# cleaning the dataset
new_df = df.dropna()  # removing the row with missing values

new_df = df.drop_duplicates()  # removing the duplicates row
print(new_df)
'''Q1) -> None are qualitative predictors'''
