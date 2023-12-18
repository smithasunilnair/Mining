#!/usr/bin/env python
# coding: utf-8

# 
# # Estimation of Used Car Prices - Data Science project
# 

# In[ ]:





# ## Import Libraries
# All libraries are used for specific tasks including data preprocessing, visualization, transformation and evaluation

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor 
import warnings
warnings.filterwarnings("ignore")


# 
# ---------------------------------------------------------
# 
# "--------------Business Problem: Estimate the price of a used car-----------"

# ## Import Data
# ### Read Training Data
# The training set is read locally and the **head** function is used to display the data for intial understanding

# "======Data understanding======"

# In[2]:


import pandas as pd

dataTrain = pd.read_csv('data_train.csv')
 
dataTrain.head()


# In[3]:


type(dataTrain)  #data type


# The **shape** function displays the number of rows and columns in the training set

# In[4]:


dataTrain.shape # check dimension


# ### Read Testing Data
# The testing set is read locally and the **head** function is used to display the data for intial understanding

# In[5]:


dataTest = pd.read_csv('data_test.csv')
dataTest.head()


# The **shape** function displays the number of rows and columns in the testing set

# In[6]:


dataTest.shape


# Checking for null values in each column and displaying the sum of all null values in each column (Training Set)

# In[7]:


dataTrain.isnull().sum()


# Checking for null values in each column and displaying the sum of all null values in each column (Testing Set)

# In[8]:


dataTest.isnull().sum()


# Removing the rows with empty values since the number of empty rows are small. This is the best approach compared to replacing with mean or random values

# In[9]:


dataTrain = dataTrain.dropna()
dataTest = dataTest.dropna()


# Checking if null values are eliminated (Training set)

# In[10]:


dataTrain.isnull().sum()


# In[11]:


dataTrain.shape # 15 rows removed


# Checking if null values are eliminated (Testing set)

# In[12]:


dataTest.isnull().sum()   


# In[13]:


dataTest.shape  # 5 rows removed


# Checking the data types to see if all the data is in correct format. All the data seems to be in their required format.

# In[14]:


dataTrain.dtypes  # checking the data type of every column


# Checking the correlation between the numerical features

# ## EDA (Exploratory Data Analysis)
# Visualizations are used to understand the relationship between the target variable and the features, in addition to correlation coefficient and p-value. 
# The visuals include heatmap, scatterplot,boxplot etc.
# 

# # Heat map

# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
corr = dataTrain.corr()  
##This is a pandas DataFrame method that is used to calculate the correlation between variables in the DataFrame.
sns.heatmap(corr,annot=True)
plt.show()


# From the heatmap, it is observed that 'year_produced' is the best feature among all the features with numerical data

# In[16]:


dataTrain.describe()  #generate various summary statistics of a DataFrame 
#Note: Only features with numeric data are considered


# A descriptive analysis to check incorrect entries and anormalies. This is also used to give an overview of the numerical data. It is observed that most of the data has no incorrect entries.

# 1. Count: The number of values in the dataframe.
# 2. Mean: The arithmetic mean or average of the values.
# 3. Standard Deviation (std): A measure of the dispersion or spread of the values.
# 4. Minimum: The minimum (smallest) value in each column.
# 5. 25th Percentile (25%): The value below which 25% of the data falls (1st quartile). Means 25% of the entire data falls under the value 158000 for odometer_value
# 6. 50th Percentile (50%): The median or value below which 50% of the data falls (2nd quartile).
# 7. 75th Percentile (75%): The value below which 75% of the data falls (3rd quartile).
# 8. Maximum: The maximum (largest) value in the Series.

# **************************************************************

# #Looking at the "minimum price", 1 USD is found.
# #This could be a wrong entry (or an outlier)
# 
# 
# 

# In[17]:


#Search for price = 1 , if so, change the price to 500
dataTrain.loc[dataTrain['price_usd'] == 1, 'price_usd'] = 500 


# In[18]:


dataTrain.describe()  # now still the minimum price is 1.42 USD


# In[19]:


#Search for price < 500 , if so, change the price to 500
dataTrain.loc[dataTrain['price_usd'] < 500, 'price_usd'] = 500


# In[20]:


dataTrain.describe()  # now the minimum price is 500 USD


# Find the distribution of the price in the entire dataset
# using "bins"  -- Technique applied is called data binning

# In[21]:


import matplotlib.pyplot as plt

dataTrain['price_usd'].plot(kind = 'hist', bins = 5, edgecolor='black')   # 5 bins are used
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Prices')
plt.show()


# From the histogram, it is understood that majority of the car samples are of lower prices

# In[22]:


dataTrain.describe(include = 'object') #summary statistics for categorical values


# ### Regression/scatter Plot
# This regression plot show the relation between **odometer** and **price**. A slight negative correlation is observed
# whaich shows that price is being affected by the change in odometer value.

# In[23]:


import seaborn as sns
plt.figure(figsize=(10,6))
sns.regplot(x="odometer_value", y="price_usd", data=dataTrain)


# As observed in the plot, a **negative correlation** is observed

# In[24]:


from scipy import stats
pearson_coef, p_value = stats.pearsonr(dataTrain['odometer_value'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# -- Pearson corr coeff of -0.42 is obtained along with a p-value of 0. 
# 
# -- The Pearson Correlation Coefficient (r) is a measure of the linear relationship between two variables. It can take values between -1 and 1
# 
# -- If r is close to 1, it indicates a strong positive linear relationship. This means that as one variable increases, the other variable tends to increase as well.
# 
# -- If r is close to -1, it indicates a strong negative linear relationship. This means that as one variable increases, the other variable tends to decrease.
# 
# -- If r is close to 0, it suggests a weak or no linear relationship. In other words, the variables are not strongly correlated.
# 
# -- Here, the Pearson Correlation Coefficient is approximately -0.422, which is closer to -1 than to 0. This indicates a moderate negative linear relationship between the two variables being correlated.
# 
# -- The p-value (probablity) is used to determine the statistical significance of the correlation. In other words, how confidently one can say a feature is correlated to the target ariable.
# 
# "IMPORTANT:" A P-value less than 0.05 (commonly used significance level) suggests that the correlation is statistically significant and hence reject the Null hypothesis. 
# 
# What is my null hypothesis? 
# H0: The feature variable is correlated to a target variable. 
# 
# Very important: A P-value of 0.0 means (more confidently say the feature is correlated to target) and that the correlation is extremely unlikely to have occurred by random chance, indicating strong statistical significance.
# 
# -- The p value here (that corresponds to odometer_values) confirms strong correlation, hence this feature is a critical feature to the prediction of used car price.

# The regression plot below shows a relationship between the year that the car is produced and the price of the car. A positive 
# correlation is observed between the two variables. This shows that the price increases with increase in production year of the car.

# In[25]:


plt.figure(figsize=(10,6))
sns.regplot(x="year_produced", y="price_usd", data=dataTrain)


# As observed above, a high positive correlation of 0.7 is calculated along with the p-value of 0. This indicates that the correlation between the variables is significant hence year produced feature can be used for prediction.

# In[26]:


pearson_coef, p_value = stats.pearsonr(dataTrain['year_produced'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# check for correlation between 'engine_capacity' and 'price'

# In[27]:


plt.figure(figsize=(10,6))
sns.regplot(x="engine_capacity", y="price_usd", data=dataTrain)


# A 0.3 correlation is calculated which is very small with a p value of 0. This indicates that even though the correlation is small but its 30% of 100 which is significant hence this feature can be used for predicition.

# In[28]:


pearson_coef, p_value = stats.pearsonr(dataTrain['engine_capacity'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# This regression plot shows an minor positive correlation observed with the help of the best fit line. The calculation will confirm the actual value.

# -----check for correlation between 'number of photos' and 'price'------------

# In[29]:


plt.figure(figsize=(10,6))
sns.regplot(x="number_of_photos", y="price_usd", data=dataTrain)


# The correlation is 0.31 based on the calculation while the p-value calculated is zero. This is similar to the last feature hence the significant 31% of 100 correlation makes this feature eligble for prediction.

# In[30]:


pearson_coef, p_value = stats.pearsonr(dataTrain['number_of_photos'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# This plot shows correlation with points all over the graph like the previous feature varibale.

# -------check correlation b/w number of mantenance and price-------------

# In[31]:


plt.figure(figsize=(10,6))
sns.regplot(x="number_of_maintenance", y="price_usd", data=dataTrain)


# The calculation proves that a correlation is lesser than 0.1 percent and indicates no correlation and the p-value lesser than 0.05 confirms it. This feature is not a critical feature for predicition
# 
# A P-value less than 0.05 (commonly used significance level) suggests that the correlation is statistically significant and hence reject the Null hypothesis. 
# 
# What is my null hypothesis? 
# H0: The number_of_maintenance is correlated to price. 
# 
# My alternate hypothesis
# HA: The number_of_maintenance is not correlated to price. 

# In[32]:


pearson_coef, p_value = stats.pearsonr(dataTrain['number_of_maintenance'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# ---- this plot shows no correlation with points all over the graph ----

# *************check correlation between duration listed and price***************

# In[33]:


plt.figure(figsize=(10,6))
sns.regplot(x="duration_listed", y="price_usd", data=dataTrain)

The calculated correlation is lesser than 0.1 which is considered negligible. The p-value lesser than 0.05 confirming the rejection of null hypothesis and hence this feature is not suitable for prediction of price. 
# In[34]:


pearson_coef, p_value = stats.pearsonr(dataTrain['duration_listed'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# ### Box Plot
# These plots are used for categorical data to determine the importance of features for prediction. 

# In the given plot below, it is observed that the price range vary for automatic and manual transmisson. This indicates the categories can vary with price hence feature can be used for prediction

# In[35]:


sns.boxplot(x="transmission", y="price_usd", data=dataTrain)


# The box plot shows how prices vary based on different colors. This shows that color can be used as a feature for price prediction.

# In[36]:


plt.figure(figsize=(10,6))
sns.boxplot(x="color", y="price_usd", data=dataTrain)


# This plot shows engine fuel types and how they affect the price. Hybrid petroll with the highest price range while hybrid diesel with lowest price range. This feature can be used for prediction.

# In[37]:


sns.boxplot(x="engine_fuel", y="price_usd", data=dataTrain)


# The engine type (based on fuel type) shows that both categories have almost the same price range which will not bring differences in price when prediction is made. Hence this feature is not suitable for price prediction

# In[38]:


sns.boxplot(x="engine_type", y="price_usd", data=dataTrain)


# Thee box plot shows body type categories with varying prices per category hence this feature can be used for price prediction, not so signficant though

# In[39]:


plt.figure(figsize=(10,6))
sns.boxplot(x="body_type", y="price_usd", data=dataTrain)


# Has warranty feature shows a huge difference in price ranges between cars with warrant and vice versa. This feature is very important for price prediction as the bigger the difference in range the better the feature.

# In[40]:


sns.boxplot(x="has_warranty", y="price_usd", data=dataTrain)


# This feature is similar to the feature above, all three categories have wider price ranges between one another. This feature is also crucial for price prediction.

# In[41]:


sns.boxplot(x="ownership", y="price_usd", data=dataTrain)


# Front and rear drive have **minimal price difference** while all drive shows a **greater difference** hence the feature can be used for prediction.

# In[42]:


sns.boxplot(x="type_of_drive", y="price_usd", data=dataTrain)


# With not same price range between categories this feature is  suitable for prediction.

# In[43]:


sns.boxplot(x="is_exchangeable", y="price_usd", data=dataTrain)


# This plot shows that the manufacturer name is not important when selling a car. The variety of price ranges for all categories prove that the feature is insignificant for price prediction.

# In[44]:


plt.figure(figsize=(10,6))
sns.boxplot(x="manufacturer_name", y="price_usd", data=dataTrain)


# Using Exploratory data analysis, few features can be dropped because they had no impact on the price prediction. Those features are removed with the function below.(Training set)

# In[45]:


dataTrain.drop(['number_of_maintenance', 'duration_listed', 'engine_type','is_exchangeable'], axis = 1, inplace = True)


# Same features are removed for testing set since the data will be used to train the model

# In[46]:


dataTest.drop(['number_of_maintenance', 'duration_listed', 'engine_type','is_exchangeable'], axis = 1, inplace = True)


# In[47]:


dataTrain.shape


# In[48]:


dataTest.shape


# ### Data Transformation
# Label encoding of categorical features in the training set. Label encoding is converting categorical data into numerical data since the model cant understand textual data.

# ----Data Preparation--------

# In[51]:


from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
dataTrain.manufacturer_name = labelencoder.fit_transform(dataTrain.manufacturer_name)
dataTrain.transmission = labelencoder.fit_transform(dataTrain.transmission)
dataTrain.color = labelencoder.fit_transform(dataTrain.color)
dataTrain.engine_fuel = labelencoder.fit_transform(dataTrain.engine_fuel)

dataTrain.body_type = labelencoder.fit_transform(dataTrain.body_type)
dataTrain.has_warranty = labelencoder.fit_transform(dataTrain.has_warranty)
dataTrain.ownership = labelencoder.fit_transform(dataTrain.ownership)
dataTrain.type_of_drive = labelencoder.fit_transform(dataTrain.type_of_drive)


# Label encoding of all categorical data in the testing set.

# In[52]:


labelencoder1 = LabelEncoder()
dataTest.manufacturer_name = labelencoder1.fit_transform(dataTest.manufacturer_name)
dataTest.transmission = labelencoder1.fit_transform(dataTest.transmission)
dataTest.color = labelencoder1.fit_transform(dataTest.color)
dataTest.engine_fuel = labelencoder1.fit_transform(dataTest.engine_fuel)

dataTest.body_type = labelencoder1.fit_transform(dataTest.body_type)
dataTest.has_warranty = labelencoder1.fit_transform(dataTest.has_warranty)
dataTest.ownership = labelencoder1.fit_transform(dataTest.ownership)
dataTest.type_of_drive = labelencoder1.fit_transform(dataTest.type_of_drive)


# Checking on the remaining features and if label encoding is applied to all categorical features (Training set).

# In[53]:


dataTrain.head(10)


# Check on the remaining features and application of label encoding to all categorical features (Testing set).

# In[54]:


dataTest.head(10)


# --Data Transfornation (normalization) ----
# z-score used for scaling down the features between the range of -1 and 1. This helps the model make better prediction as it is easy to understand. The scaling is applied to the training and testing set  --- You can try using min-max normalization also

# In[55]:


# Calculate the z-score from with scipy
import scipy.stats as stats
dataTrain = stats.zscore(dataTrain)
dataTest = stats.zscore(dataTest)


# In[56]:


dataTrain


# In[57]:


dataTest


# Dividing the data for training and testing accordingly. X takes the all features while Y takes the target variable
# 
# We have 13 actual columns [0-12 index]; 12 are predictor variables and 1 is the target variable

# In[58]:


x_train=dataTrain.iloc[:,0:11]
y_train=dataTrain.iloc[:,12]
x_test=dataTest.iloc[:,0:11]
y_test=dataTest.iloc[:,12]


# In[59]:


x_train.head()


# In[60]:


y_train.head()


# ## Fit Model
# ### Multiple Linear Regression
# Calling multiple linear regression model and fitting the training set

# In[61]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model_mlr = model.fit(x_train,y_train)


# Making price prediction using the testing set (Fit to MLR)

# In[62]:


y_pred1 = model_mlr.predict(x_test)


# In[63]:


#randomly checking the y-test values 
y_test[0]


# In[64]:


#randomly checking the y-pred values 
y_pred1[0]


# y_test[0]   and   y_pred1[0]   have different values.. In other words, there is error

# ### MLR Evaluation
# 

# Calculating the Mean Square Error for MLR model

# In[65]:


mse1 = mean_squared_error(y_test, y_pred1)
print('The mean square error for Multiple Linear Regression: ', mse1)


# Calculating the Mean Absolute Error for MLR model

# In[66]:


mae1= mean_absolute_error(y_test, y_pred1)
print('The mean absolute error for Multiple Linear Regression: ', mae1)


# ### Random Forest Regressor (checking other Models)
# Calling the random forest model and fitting the training data

# In[67]:


rf = RandomForestRegressor()
model_rf = rf.fit(x_train,y_train)


# Prediction of car prices using the testing data

# In[68]:


y_pred2 = model_rf.predict(x_test)


# ### Random Forest Evaluation
# 

# Calculating the Mean Square Error for Random Forest Model (Lowest MSE value)

# In[69]:


mse2 = mean_squared_error(y_test, y_pred2)
print('The mean square error of price and predicted value is: ', mse2)


# Calculating the Mean Absolute Error for Random Forest Model (Lowest Mean Absolute Error)

# In[70]:


mae2= mean_absolute_error(y_test, y_pred2)
print('The mean absolute error of price and predicted value is: ', mae2)


# ### LASSO Model 
# Calling the model and fitting the training data

# In[71]:


LassoModel = Lasso()
model_lm = LassoModel.fit(x_train,y_train)


# Price prediction uisng testing data

# In[72]:


y_pred3 = model_lm.predict(x_test)


# ### LASSO Evaluation  (checking another model)
# 

# Mean Absolute Error for LASSO Model

# In[73]:


mae3= mean_absolute_error(y_test, y_pred3)
print('The mean absolute error of price and predicted value is: ', mae3)


# Mean Squared Error for the LASSO Model

# In[74]:


mse3 = mean_squared_error(y_test, y_pred3)
print('The mean square error of price and predicted value is: ', mse3)


# In[75]:


scores = [('MLR', mae1),
          ('Random Forest', mae2),
          ('LASSO', mae3)
         ]         


# In[76]:


mae = pd.DataFrame(data = scores, columns=['Model', 'MAE Score'])
mae


# In[78]:


mae.sort_values(by=(['MAE Score']), ascending=False, inplace=True)

f, axe = plt.subplots(1,1, figsize=(10,7))
sns.barplot(x = mae['Model'], y=mae['MAE Score'], ax = axe)
axe.set_xlabel('Model', size=20)
axe.set_ylabel('Mean Absolute Error', size=20)

plt.show()


# #Based on the MAE, it is concluded that the Random Forest is the best regression model for predicting the car price based on the 12 predictor variables 

# In[ ]:




