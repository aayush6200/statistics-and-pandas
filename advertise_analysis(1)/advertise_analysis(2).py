import pandas as pd
import statistics
from scipy.stats import gmean

import matplotlib.pyplot as plt
# this program focuses on handling any duplicates, unnecessary data: cleaning data

r_path = './dataset/advertise_data.csv'


data = pd.read_csv(r_path)


# returns the count of missing values for each column in the DataFrame.
missed_in_columns = data.isnull()

# print(missed_in_columns)    ## prints true or false depending whether anything missed

data_cleaned = data.dropna()  # cleans the missing values in rows

# print(data_cleaned)   ## prints a new dataset with cleaned data

# returns the total no of duplicates in cleaned data
total_duplicates_in_cleaned_data = data.duplicated().sum()

# print(total_duplicates_in_cleaned_data) ## prints the total number of duplicates


# new data without duplicates data
new_dataset_without_duplicates = data.drop_duplicates()
# prints the new dataset without duplicates
print(new_dataset_without_duplicates)

# below program focuses on data exploration using mean, median  and standard deviation and their analysis

tv_mean_AM = new_dataset_without_duplicates['TV'].mean()

# print(tv_mean_AM)  # prints Arithmetic mean for TV columns

tv_mean_GM = gmean(new_dataset_without_duplicates['TV'])

# print(tv_mean_GM)  # prints Geometric mean for TV columns


row_1_mean = new_dataset_without_duplicates.loc[1].mean()
# print(row_1_mean)  # prints mean for 1st row


# calculate median::


tv_column_median = new_dataset_without_duplicates['TV'].median()


print('median tv', tv_column_median)  # prints the median of TV column


# calculate standard deviation


tv_column_sd = new_dataset_without_duplicates['TV'].std()


print('standard deviation', tv_column_median)  # prints standard deviation


# extracting data
tv_expense = data['TV']
sales = data['sales']

# plotting data

# plt.plot(tv_expense, sales, 'o')


# plt.xlabel('TV Ad EXPENSE')
# plt.ylabel('Sales')

# plt.title('TV Ad Expense vs Sales')
# plt.grid(True)
# plt.show()


# You can adjust the number of bins as per your preference
# plt.hist(data['TV'], sales)
# plt.xlabel('TV Budget')
# plt.ylabel('Frequency')
# plt.title('Histogram of TV Budget')
# plt.show()


# correlation analysis between sales and TV


correlation_tv_sales = data['TV'].corr(data['sales'])


print('The correlation between sales and TV is', correlation_tv_sales)
