import pandas as pd


r_path='./dataset/advertise_data.csv'  ## relative path of csv model


data=pd.read_csv(r_path)


print(data.head())  ## gives first few  data
print(data.tail())   ## gives last few data

print(data)  ## presents data


