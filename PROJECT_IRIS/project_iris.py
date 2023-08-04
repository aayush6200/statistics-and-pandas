import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
r_path = './dataset/iris/iris.csv'

# converting csv into dataset
iris_df = pd.read_csv(r_path)

print('This is iris dataset\n', iris_df)


# cleaning up our dataset

iris_df = iris_df.dropna()  # cleaning up the row if its missing


iris_df = iris_df.drop_duplicates()  # removing the duplicates row


print('The cleaned iris dataset is \n', iris_df)


# shuffle thd data
shuffled_df = iris_df.sample(frac=1, random_state=42)

train_size = int(0.7*len(shuffled_df))  # training data size

train_df = shuffled_df.iloc[:train_size]  # dataset for training our model
train_df.to_csv('train.csv', index=False)  # saving our dataset on train.csv


test_df = shuffled_df.iloc[train_size:]

test_df.to_csv('test.csv', index=False)  # saving our dataset on test
print('hello world')


# training our data using linear regression

# reading csv files for training and testing
train_df = pd.read_csv('./dataset/iris/train.csv')
test_df = pd.read_csv('./dataset/iris/test.csv')


# Create a LabelEncoder object
label_encoder = LabelEncoder()
train_df['species'] = label_encoder.fit_transform(train_df['species'])
test_df['species'] = label_encoder.transform(test_df['species'])
x_train = train_df[['sepal_length', 'sepal_width',
                    'petal_length', 'petal_width']].values
y_train = train_df['species'].values

x_test = test_df[['sepal_length', 'sepal_width',
                  'petal_length', 'petal_width']].values
y_test = test_df['species'].values

# create the model
print("Number of samples in X_train:", x_train.shape[0])
print("Number of samples in y_train:", y_train.shape[0])
print("Number of samples in X_test:", x_test.shape[0])
print("Number of samples in y_test:", y_test.shape[0])

model = LinearRegression()
model.fit(x_train, y_train)

# lets predict now
input_data = [[5.6, 2.5, 3.9, 1.1]]
y_predict = model.predict(x_test)

predicted_names = label_encoder.inverse_transform(
    y_predict.astype(int))
# mse = mean_squared_error(y_test, y_predict)
# rmse = mse**0.5
# print('tHE MSE IS ', mse, rmse)
print('The predicted flower species are:', predicted_names)

# analyzing our data by plotting in graph

# for sepal_length vs species

# sepal_length = iris_df['sepal_length']  # sepal_length in the single dataFrame
# sepal_width = iris_df['sepal_width']  # sepal width in dataFrame
# petal_length = iris_df["petal_length"]  # petal length in dataFrame
# petal_width = iris_df["petal_width"]  # petal width in dataFrame
# # putting all the species into the single dataFrame
# species = iris_df['species']


# # plotting sepal_length vs species graphs representing with 'x
# plt.plot(sepal_length, species, 'x')


# plt.xlabel('Sepal Length')
# plt.ylabel('species')

# plt.show()


# # for sepal_width vs species


# plt.plot(sepal_width, species, 'x')  # plotting the graph

# plt.xlabel('Sepal width')
# plt.ylabel('Species')

# plt.show()


# # for petal_length

# plt.plot(petal_length, species, 'x')  # plotting the graph
# plt.xlabel('Petal length')
# plt.ylabel('Species')
# plt.show()


# # for petal_width

# plt.plot(petal_width, species, 'x')  # plotting the graph

# plt.xlabel('Petal width')
# plt.ylabel('Species')
# plt.show()
# plot the data points and the linear regression line
x_train = x_train.reshape(-1, 1)
plt.scatter(x_train, y_train, color='blue', label='Data Points')
plt.plot(x_train, model.predict(x_train),
         color='red', label='Linear Regression Line')
plt.xlabel('Sepal Length')
plt.ylabel('Species (Encoded)')
plt.legend()
plt.show()
