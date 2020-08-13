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
    df = pd.read_excel('population final.xlsx')
    # df = df[df['country'] == 'INDIA']


    years = np.array([1901, 1911, 1921, 1931, 1941, 1951, 1961, 1971, 1981, 1991, 2001, 2011])
    years = np.reshape(years, (12, 1))
    years_tbp = np.array([[2021], [2031], [2041], [2051]])

    for country in set(df['country']):
        #get the part of the dataframe for a particular state
        #Includes the full prediction of India as well, because INDIA is a part
        #the country coloumn
        df_country = df[df['country'] == country].reset_index(drop = True)
        if len(df_country) != 12 or 'N.A' in list(df_country['persons']):
            continue

        #get population data
        data = np.array(df_country['persons'])
        data = np.reshape(data, (12, 1))

        #see ref
        polynomial_feature = PolynomialFeatures(degree = 3)
        x_poly = polynomial_feature.fit_transform(years)

        model = LinearRegression()
        model.fit(x_poly, data)
        test_preds = model.predict(x_poly)

        x_future_poly = polynomial_feature.fit_transform(years_tbp)
        future_preds = model.predict(x_future_poly)

        mse = mean_squared_error(data, test_preds)
        print(f'mse for {country} : {mse:.2e}')

        plt.clf()
        plt.scatter(years, data, label = 'ground truth')
        plt.scatter(years, test_preds, color = 'red', label = 'test predicted')
        plt.scatter(years_tbp, future_preds, color = 'green', label = 'future predicted')
        plt.title(country.translate(str.maketrans('', '', string.punctuation)))
        plt.legend()
        plt.xlabel('years')
        plt.ylabel('population')
        plt.show()
