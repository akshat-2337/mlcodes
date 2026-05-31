import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import preprocessing, linear_model

# Note the double backslashes (\\) to keep Windows from getting confused
df = pd.read_csv("C:\\python vsc\\Multiple Linear Regression\\FuelConsumptionCo2.csv")
#loading the Fuel Consumption dataset

df.sample(5)
#loading 5 random sample data

df.describe()
#getting parameters/data which can help further

df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1)
#removing unessecary parameters for modelling illustration

df.corr()
#correlation matrix shows pairwise correlation b/w all features 
#it also indicates how predictive each feature is of the target.

df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',],axis=1)
#dropping more unessecary parameters and keeping only the ones which are most useful 

df.head(9)

axes = pd.plotting.scatter_matrix(df, alpha=0.2)
# need to rotate axis labels so we can read them
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()
#scatter plot of each pair of input features to basically help in selective the predictive features.    

x = df.iloc[:,[0,1]].to_numpy()
y = df.iloc[:,[2]].to_numpy()
#extracting required columns which is needed for MLR model

std_scaler = preprocessing.StandardScaler()
x_std = std_scaler.fit_transform(x)
#preprocessing the data so that the model does not favour any outlying/irrelevant data due to the magnitude

x_test, x_train, y_test, y_train = train_test_split(x_std, y, test_size=0.2, random_state=42)
#creating test train split with preprocessed data i.e x_std and y

mlr = linear_model.LinearRegression()
mlr.fit(x_train, y_train)
#creating the model object and training the model using the training data

coef = mlr.coef_
intr = mlr.intercept_
#getting the coefficient and intercept of the model

means_ = std_scaler.mean_
std_devs_=np.sqrt(std_scaler.var_)
#calculating the standard mean and standard deviation

coef_original = coef/std_devs_
intr_original = intr-np.sum((means_*coef)/std_devs_)

#To convert a machine learning model trained on standardized data back into a normal, 
# human-readable equation. [Lines 61-66]

x1 = x_test[:,0] if x_test.ndim > 1 else x_test
x2 = x_test[:,1] if x_test.ndim > 1 else np.zeros_like[x1]
#x1 is first column x2 is 2nd column
#if there is only 1 feature x1 will be as it is (first column) x2 will be same size as x1 with all zeros.

x1_surf, x2_surf = np.meshgrid(np.linspace(x1.min(), x1.max(), 100),
                               np.linspace(x2.min(), x2.max(), 100))
#creating meshgrid with linspace as parameters.
#linspace : Create evenly spaced numbers between two limits.
#linspace limit is from x.min to x.max and there are 100 of those numbers
#meshgrid : create all possible coordinate combination values from x and y

y_surf = intr + coef[0,0] * x1_surf + coef[0,1] * x2_surf
#equation of multi linear regression.
#x1_surf and x2_surf are features here.

y_pred = mlr.predict(x_test)
#predicting y values using trained regressor model.

above_plane = y_test>=y_pred
below_plane = y_test<y_pred
above_plane = above_plane[:,0]
below_plane = below_plane[:,0]

fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1[above_plane], x2[above_plane], y_test[above_plane],  label="Above Plane",s=70,alpha=.7,ec='k')
ax.scatter(x1[below_plane], x2[below_plane], y_test[below_plane],  label="Below Plane",s=50,alpha=.3,ec='k')

ax.plot_surface(x1_surf, x2_surf, y_surf, color='k', alpha=0.21, label="Plane")

ax.view_init(elev=10)

ax.legend(fontsize="x-large",loc="upper center")
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(None, zoom=0.75)
ax.set_xlabel('ENGINESIZE' , fontsize="xx-large")
ax.set_ylabel("FUEL CONSUMPTION", fontsize="xx-large")
ax.set_zlabel("CO2 EMISSION", fontsize="xx-large")
ax.set_title("MLR OF CO2 EMISSIONS", fontsize="xx-large")
plt.tight_layout()
plt.show()
