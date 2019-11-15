"""
Data Mining - Advanced Statistical Modeling
Linear Regression
Predicting the Price of Diamonds

Siphu Langeni
"""


import os
os.chdir('/Volumes/ABE2017/Python Practice')
os.chdir(r'D:\Python Practice')
os.getcwd()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import csv
Diamonds = pd.read_csv('Diamonds.csv')

Diamonds.info()
Diamonds.isnull().sum() # no missing values
Diamonds.describe()

matrix = np.triu(Diamonds.corr(), 1)
plt.figure(figsize=(19, 15))
sns.heatmap(Diamonds.corr(), annot = True, square = True, mask = matrix, linewidths = 0.5, annot_kws = {'size': 16})
plt.title('Correlation Matrix', fontsize = 20)
plt.tick_params(labelsize = 18)
plt.show()

Diamonds.drop(['depth', 'table'], axis = 1, inplace = True)



Diamonds.cut = Diamonds.cut.map(
                                {'Fair': 0,
                                 'Good': 1,
                                 'Very Good': 2,
                                 'Premium': 3,
                                 'Ideal': 4
                                 }
                                )

Diamonds.color = Diamonds.color.map(
                                    {'J': 0,
                                     'I': 1,
                                     'H': 2,
                                     'G': 3,
                                     'F': 4,
                                     'E': 5,
                                     'D': 6
                                     }
                                    )

Diamonds.clarity = Diamonds.clarity.map(
                                        {'I1': 0,
                                         'SI2': 1,
                                         'SI1': 2,
                                         'VS2': 3,
                                         'VS1': 4,
                                         'VVS2': 5,
                                         'VVS1': 6,
                                         'IF': 7
                                         }
                                        )


from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Regress carat against all other features
y, X = dmatrices('carat ~ cut + color + clarity + x + y + z', Diamonds, return_type = 'dataframe')

# For each X, calculate VIF and save in dataframe
VIF = pd.DataFrame()
VIF['Features'] = X.columns
VIF['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(VIF)


Diamonds.drop(['x', 'y', 'z'], axis = 1, inplace = True)

# Establish in- and dependent variables
X = Diamonds.drop(['price'], axis = 1)
y = Diamonds.price

from sklearn.model_selection import train_test_split
# Create train and test sets 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 313)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)
print('R2: %0.5f' % lr.score(X_test, y_test))

# Actual vs Predicted values of y
plt.scatter(y_test, lr_y_pred, s = 3)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices',fontsize = 20)

from sklearn.preprocessing import PolynomialFeatures
# Determine best number of polynomial degrees
R2 = []
for i in range(1, 8):
    poly_feat = PolynomialFeatures(degree = i)
    X_poly_train = pd.DataFrame(poly_feat.fit_transform(X_train))
    X_poly_test = pd.DataFrame(poly_feat.transform(X_test))
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)
    y_pred = poly_reg.predict(X_poly_test)
    R2.append(r2_score(y_pred, y_test))
print('Highest R2 =', round(max(R2), 3), 'occurs when degree =', R2.index(max(R2)) + 1,'.')

# Plot



# Using 5 degrees
poly_feat = PolynomialFeatures(degree = 5)
X_poly_train = pd.DataFrame(poly_feat.fit_transform(X_train))
X_poly_test = pd.DataFrame(poly_feat.transform(X_test))
poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)
poly_y_pred = poly_reg.predict(X_poly_test)

print('MODEL METRICS (Train)', '\nR2: %0.5f' % poly_reg.score(X_poly_train, y_train), '\nAdj R2: %0.5f' % (1 - (((len(X_poly_test.index) - 1) / (len(X_poly_test.index) - len(X_poly_test.columns) - 1)) * (1 - poly_reg.score(X_poly_train, y_train)))), '\nRMSE: $%0.0f' % np.sqrt(mean_squared_error(y_train, poly_reg.predict(X_poly_train))), '\n\nMODEL METRICS (Test)', '\nR2: %0.5f' % poly_reg.score(X_poly_test, y_test), '\nAdj R2: %0.5f' % (1 - (((len(X_poly_test.index) - 1) / (len(X_poly_test.index) - len(X_poly_test.columns) - 1)) * (1 - poly_reg.score(X_poly_test, y_test)))), '\nRMSE: $%0.0f' % np.sqrt(mean_squared_error(y_test, poly_y_pred)))


from sklearn.linear_model import Ridge
# Add some regularization
poly_reg = Ridge(alpha = 10)
poly_reg.fit(X_poly_train, y_train)

print('MODEL METRICS (Train)', '\nR2: %0.5f' % poly_reg.score(X_poly_train, y_train), '\nAdj R2: %0.5f' % (1 - (((len(X_poly_test.index) - 1) / (len(X_poly_test.index) - len(X_poly_test.columns) - 1)) * (1 - poly_reg.score(X_poly_train, y_train)))), '\nRMSE: $%0.0f' % np.sqrt(mean_squared_error(y_train, poly_reg.predict(X_poly_train))), '\n\nMODEL METRICS (Test)', '\nR2: %0.5f' % poly_reg.score(X_poly_test, y_test), '\nAdj R2: %0.5f' % (1 - (((len(X_poly_test.index) - 1) / (len(X_poly_test.index) - len(X_poly_test.columns) - 1)) * (1 - poly_reg.score(X_poly_test, y_test)))), '\nRMSE: $%0.0f' % np.sqrt(mean_squared_error(y_test, poly_y_pred)), '\n\nK-FOLD CROSS VALIDATION', '\nR2: %0.5f' % cross_val_score(Ridge(alpha = 10), X_poly_train, y_train, cv = 4).mean(), '(k = 4)')












# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)




# Train metrics
np.sqrt(mean_squared_error(y_train, pol_reg.predict(poly_reg.fit_transform(X_train)))) #train RMSE
r2_score(y_train, pol_reg.predict(poly_reg.fit_transform(X_train))) # train r2
1 - (((len(X_poly_train.index) - 1) / (len(X_poly_train.index) - len(X_poly_train.columns) - 1)) * (1 - R2_train))

# Test metrics
np.sqrt(mean_squared_error(y_test, pol_reg.predict(poly_reg.fit_transform(X_test)))) #test RMSE
r2_score(y_test, pol_reg.predict(poly_reg.fit_transform(X_test)))

# Test the model
y_pred = poly_reg.predict(X_test)

# y-intercept
y_int = model.intercept_

# Coefficients for all the features in the model
coeff = pd.DataFrame(model.coef_, x.columns, columns = ['Coefficient'])

# Metrics for the model
MSE = mean_squared_error(y_test, y_pred) # MSE test
RMSE = np.sqrt(MSE) #RMSE test
R2_test = r2_score(y_test, y_pred) # R2 test
AR2_test = 1 - (((len(x_test.index) - 1) / (len(x_test.index) - len(x_test.columns) - 1)) * (1 - R2_test)) # Adjusted R2 test

# Metrics for the training data
MSE = mean_squared_error(y_train, model.predict(x_train))
RMSE_train = np.sqrt(MSE)
R2_train = r2_score(y_train, model.predict(x_train))
AR2_train = 1 - (((len(x_train.index) - 1) / (len(x_train.index) - len(x_train.columns) - 1)) * (1 - R2_train)) # Adjusted R2 train
# K-Fold using 3 folds
K_Fold_3 = cross_val_score(LinearRegression(), x_train, y_train, cv = 3).mean()

# Boxplot - Price
box_plot_data = y.tolist()
plt.boxplot(box_plot_data)
plt.title('BOX PLOT FOR\nDIAMOND PRICES')
plt.show()
bp = plt.boxplot(box_plot_data, patch_artist = True, labels=['Diamond Prices'])



Pearson = stats.pearsonr(Diamonds.price, Diamonds.table)[0]
Pearson = stats.pearsonr(Diamonds.price, Diamonds.depth)[0]
Pearson[0] ** 2
# Carat vs Price
plt.scatter(x_test.carat, y_test, c = 'blue', marker = '*')
plt.xlabel('Carat Weight')
plt.ylabel('Price (USD)')
plt.title('Price of Diamonds',fontsize = 20)








'''A data frame with 53940 rows and 10 variables:

price
price in US dollars (\$326–\$18,823)

carat
weight of the diamond (0.2–5.01)

cut
quality of the cut (Fair, Good, Very Good, Premium, Ideal)

color
diamond colour, from J (worst) to D (best)

clarity
a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

x
length in mm (0–10.74)

y
width in mm (0–58.9)

z
depth in mm (0–31.8)

depth
total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43–79)

table
width of top of diamond relative to widest point (43–95)
'''










