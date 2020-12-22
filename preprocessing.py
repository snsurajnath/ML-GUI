######## Data preprocessing

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#importing dataset
dataset = pd.read_csv('Data.csv')
# x is taking all feature data i.e. independent variables
x = dataset.iloc[ : , : -1].values # include all rows except the last result column
# y is taking all dependent data i.e. result column
y= dataset.iloc[ : , -1].values # getting last column of the data

print(x)
print(y)

# taking care of missing data in dataset
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[ : , 1: 3]) # fit method checks all missing values in dataset and after we have to apply transform method
x[ : , 1:3] = imputer.transform(x[ : , 1:3]) # transform method charges that missing values by mean of all
# x[ : , 1:3] update the missing value in dataset
print(x)

## encoding independent string variables
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)
## encoding dependent string variables

le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[: , 3: ] = sc.fit_transform(x_train[: , 3: ])
x_test[: , 3: ] = sc.transform(x_test[: , 3: ])

print(x_train)
print(x_test)