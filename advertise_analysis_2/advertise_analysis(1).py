from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd

r_path = './dataset/advertise_data.csv'

df = pd.read_csv(r_path)

# cleaning up the dataset

df = df.dropna()  # removes the row with the missing data

cleaned_df = df.drop_duplicates()  # removes the duplicates row

print(cleaned_df)  # cleaned dataset


# shuffle dataset

# shuffles whole row of dataset
shuffled_df = cleaned_df.sample(frac=1, random_state=42)

split_index = int(0.5 * len(shuffled_df))

# creates the dataset for training our model
train_df = shuffled_df.iloc[:split_index]

# creates the dataset for testing our model
test_df = shuffled_df.iloc[split_index:]


# save test and training files as csv files in root directory


train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)


# checking if datasets have been saved


r1_path = './dataset/train_data.csv'

# print(pd.read_csv(r1_path)) verified


# building linear regression model from train_data


train_df = pd.read_csv(r1_path)  # reading data to train

x_train = train_df["TV"].values.reshape(-1, 1)
y_train = train_df["sales"].values
model = LinearRegression()
model.fit(x_train, y_train)

# for testing data
y_pred = model.predict([[296.4]])
print("Predicted sales:", y_pred[0])
MSE = mean_squared_error(y_pred, [23.8])
print("Mean Squared Error:", MSE)  # got 8.2
