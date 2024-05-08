import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the advertising data
advertising_data = pd.read_csv('Advertising.csv', skiprows=1, names=['Index', 'TV', 'Radio', 'Newspaper', 'Sales'], encoding='ascii')

# Display the first few rows of the dataframe
print(advertising_data.head())

# Set the aesthetic style of the plots
sns.set(style='whitegrid')

# Descriptive statistics
print(advertising_data.describe())

# Pairplot to visualize the relationships between features
plt.figure(figsize=(10, 8))
sns.pairplot(advertising_data, kind='reg', plot_kws={'line_kws':{'color':'red'}})
plt.show()

# Compute the correlation matrix
correlation_matrix = advertising_data.corr()
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Selecting features and target
X = advertising_data[['TV', 'Radio']]
y = advertising_data['Sales']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Calculating R-squared and RMSE
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print('R-squared:', r2)
print('RMSE:', rmse)
