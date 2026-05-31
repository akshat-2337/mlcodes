import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("FuelConsumptionCo2.csv")
#reading dataset using pandas

print(df.sample(5))
#printing random 5 entries for showing dataset is loaded and working fine

print(df.describe())
#exploring the data

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#data keys which might be indicators for CO2 emissions.
print(cdf.sample(5))
cdf.hist()
plt.show()

#enginesize vs emissions
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#fuel vs emissions
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='green')
plt.xlabel("Fuel consumption")
plt.ylabel("Emission")
plt.show()

#cylinder vs emission - q1
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()

x = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

x_train , x_test , y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#test train split 

regressor = linear_model.LinearRegression()
#creating model object

regressor.fit(x_train.reshape(-1,1),y_train)
#calling the linear regression function; this is the function where the model learns from.

print("Coefficients = ", regressor.coef_[0])
print("Intercept = ",regressor.intercept_)
#prints the Coefficient and Intercept of the regressor, since it's a simple linear regressor there is only 1 coefficient

plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train, regressor.coef_*x_train+regressor.intercept_,color="red")
plt.xlabel("engine size")
plt.ylabel("co2 emission")
plt.title("MODEL OUTPUT")
plt.show()
#plotting the model output;
#First a scatter plot showing all the training dataset's relation on x and y axis.
#Second a straight line plotted on the same graph with the formula y=mx+c. (Line of best fit).

y_pred = regressor.predict(x_test.reshape(-1,1))
#regressor.predict is used to make test predictions by taking in the x_test in the parameters and reshaping it to -1,1

print("Mean Absolute Error = ", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error = ", mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error = ", np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2 Score = ", r2_score(y_test,y_pred))
#Printing various model evaluation metrics by passing y_test and y_pred as parameters.
#Mean Absolute Error (MAE):** The average horizontal/vertical distance between the model's predictions and the actual data points, showing how far off the guesses are on average in the same units as the target.
#Mean Squared Error (MSE):** The average of the squared differences between predicted and actual values, which punishes larger prediction mistakes much more heavily than smaller ones.
#R-squared ($R^2$) Score:** A metric from 0 to 1 (or below) that measures the proportion of variance in the target variable that the model successfully explains, essentially grading how much better the model is than just guessing the average.

plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test, regressor.coef_*x_test+regressor.intercept_,color="red")
plt.xlabel("engine size")
plt.ylabel("emission")
plt.title("MODEL OUTPUT - TEST DATA")
plt.show()
#plotting the model result over the test data instead of the training data for experimental purposes.

x = cdf.FUELCONSUMPTION_COMB_MPG.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#creating data set for a new model which is used for MPG vs CO2 Emission

regr = linear_model.LinearRegression()
#creating model object

regr.fit(x_train.reshape(-1,1),y_train)
#calling the .fit() function from which the model learns

coef = regr.coef_[0]
intr = regr.intercept_
#getting the coefficient and intercept for the model.

plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train, coef*x_train+intr,color="red")
plt.xlabel("mpg")
plt.ylabel("co2 emission")
plt.title("MODEL OUTPUT - mpg vs co2")
plt.show()
#scatter plot of mpg vs co2 emission against model evaluated line.

y_pred = regr.predict(x_test.reshape(-1,1))
#creating prediction dataset

print("Mean Squared Error = ", mean_squared_error(y_test, y_pred))
#printing mean squared error for mpg vs co2 emission model.