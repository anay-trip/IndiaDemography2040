import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

'''
ref
https://towardsdatascience.com/polynomial-regression-bbe8b9d97491
'''

if __name__ == '__main__':
    f = open('lifeExp.csv', 'r')

    years = np.array([1995, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2014])
    years = np.reshape(years, (-1, 1))
    years_tbp = np.array([[2017], [2020], [2023], [2026]])


    for line in f.readlines()[1:]:
        line = line.split(';')

        country = line[0]

        line = line[1:]

        if '' in line:
            continue

        data = np.array(line).astype(float)
        data = np.reshape(data, (-1, 1))
        polynomial_feature = PolynomialFeatures(degree = 2)
        x_poly = polynomial_feature.fit_transform(years)

        model = LinearRegression()
        model.fit(x_poly, data)
        test_preds = model.predict(x_poly)

        x_future_poly = polynomial_feature.fit_transform(years_tbp)
        future_preds = model.predict(x_future_poly)

        mse = mean_squared_error(data, test_preds)
        print (f'mse for {country} : {mse:.2e}')

        plt.clf()
        plt.scatter(years, data, label = 'ground truth')
        plt.scatter(years, test_preds, color = 'red', label = 'test predicted')
        plt.scatter(years_tbp, future_preds, color = 'green', label = 'future predicted')
        plt.title(country.translate(str.maketrans('', '', string.punctuation)))
        plt.legend()
        plt.xlabel('years')
        plt.ylabel('population')
        plt.show()
