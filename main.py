
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
# Loading the DATA ---->

data = pd.read_csv(
    'PhaDAii_2/Python/Brain_Mentor_Assignment/Covid_Cases_predictor/totalCases.csv', sep=',')
data = data[['id', 'total_cases']]
print('-'*40)
print(' '*17, 'HEAD')
print('-'*40)
print(data.head())

# Prepare DATA ----->
print('-'*40)
print(' '*17, 'PREPARE')
print('-'*40)
x = np.array(data['id']).reshape(-1, 1)
y = np.array(data['total_cases']).reshape(-1, 1)
plt.plot(y, '-m*')
# plt.show()

polyfeature = PolynomialFeatures(degree=10)
x = polyfeature.fit_transform(x)
# print(x)


# Training the DATA --->
print('-'*40)
print(' '*14, 'TRAINING DATA')
print('-'*40)

model = linear_model.LinearRegression()
model.fit(x, y)
accuracy = model.score(x, y)
print(f'Accuracy:{round(accuracy*100,3)}%')

y0 = model.predict(x)

# Prediction ------->
days = 7
print('-'*40)
print(' '*17, 'PRIDICTION')
print('-'*40)
# print(f'Pridiction after {days} days:- ',end='')
cases_after = round(
    int(model.predict(polyfeature.fit_transform([[823+days]]))))

x1 = np.array(list(range(1, 823+days))).reshape(-1, 1)
y1 = model.predict(polyfeature.fit_transform(x1))
plt.plot(y1, '--y')
plt.plot(y0, '--r')
plt.show()

# print(y[:-2:-1])
curr_cases = 509365131 # current cases till 24 apr. 2022
cases_before = int(y[:-2:-1])
case_diff = cases_after-cases_before
total_cases = curr_cases+case_diff
print(f'Total Cases on 24 Apr. 2022 is :-  {curr_cases}')
print(f'Total Cases till 30 Apr. 2022 will be :-  {total_cases}')
print(f'The Difference between Cases from (24 Apr. 2022) to (30 Apr. 2022) will be :- ',total_cases-curr_cases)

# 523688220
# print(2172179799-2151834410)