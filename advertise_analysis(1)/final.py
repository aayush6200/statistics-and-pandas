# this program focuses on finding relationships between different ad expenses and sales made


# all the findings will be written final.txt

import pandas as pd
import matplotlib.pyplot as plt

# defines the relative path to csv file
r_path = './dataset/advertise_data.csv'

# converting csv file to DataFrame
df = pd.read_csv(r_path)

# remove any missing values from row along with the row
cleaned_df = df.dropna()

# remove any duplicates from the DataFrame
duplicates_removed_df = cleaned_df.drop_duplicates()

# data is cleaned : missing values are removed, duplicate values are removed
print(duplicates_removed_df)


# lets make a scatter plot that will plot tv, radio and newspaper


# plotting TV
plt.scatter(duplicates_removed_df['TV'], duplicates_removed_df['sales'])
plt.xlabel('TV Ad Expense')
plt.ylabel('Sales')
plt.title('TV Ad Expense VS Sales')
plt.show()

# plotting radio expense
plt.scatter(duplicates_removed_df['radio'], duplicates_removed_df['sales'])
plt.xlabel('Radio Ad Expense')
plt.ylabel('Sales')
plt.title('Radio VS Sales')
plt.show()


# plotting newspaper expense
plt.scatter(duplicates_removed_df['newspaper'], duplicates_removed_df['sales'])
plt.xlabel('Newspaper Ad Expense')
plt.ylabel('Sales')
plt.title('Newspaper Ad Expense VS Sales')
plt.show()


# lets calculate the correlation between each of TV, Radio and Newspaper vs Sales

correlation_tv_sales = df['TV'].corr(df['sales'])
correlation_radio_sales = df['radio'].corr(df['sales'])
correlation_newspaper_sales = df['newspaper'].corr(df['sales'])


print('The correlation between Tv and Sales is: ', correlation_tv_sales)

print('The correlation between Radio and Sale is: ', correlation_radio_sales)

print('The correlation between newspaper and Sales is: ',
      correlation_newspaper_sales)
