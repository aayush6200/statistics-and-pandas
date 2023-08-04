import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import the college data

r_path = ('./dataset/college.csv')
# reading college data as df
df = pd.read_csv(r_path)
# df = pd.read_csv(r_path, index_col=0)
df.rename(columns={'Unnamed: 0': 'College'}, inplace=True)

# renaming
college = df

# print(college.describe())


# lets plot [Top10perc, Apps, Enroll]

matrix = np.array([df["Top10perc"], df["Apps"], df["Enroll"]])

# plotting the graphs which are histrograms and scatter plots
# print(matrix)
# pd.plotting.scatter_matrix(df[['Top10perc', 'Apps', 'Enroll']])
# plt.show()
# print(df[['Top10perc']].describe())


# Use the boxplot() method of college to produce side-by-side
# boxplots of Outstate versus Private.


df.boxplot(column='Outstate', by='Private')


# Set the title and labels
plt.title("Boxplot of Outstate by Private")
plt.xlabel("Private (0: No, 1: Yes)")
plt.ylabel("Outstate")

# Show the plot
plt.show()
