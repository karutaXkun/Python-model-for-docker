# %%
"""
Importing Libraries
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot
import matplotlib.pyplot as plt
import seaborn as sns
# %%matplotlib inline 
# %%

# %%
"""
Importing dataset
"""

# %%
dataset=pd.read_csv('project.csv')
dataset

# %%
"""
Data wrangling with X and Y
"""

# %%
x=dataset.iloc[:,0:-1].values
x

# %%
y=dataset.iloc[:,7:].values
y

# %%
"""
Checking for any NaN values
"""

# %%
dataset.isnull().any()

# %%
"""
Training and Testing by using Scikit Learn
"""

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# %%
"""
importing LinearRegression from Scikit Learn model
"""

# %%
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

# %%
"""
Fitting x and y values
"""

# %%
lr.fit(x_train,y_train)

# %%
"""
Predicting The y values
"""

# %%
y_predict=lr.predict(x_test)
y_predict

# %%
lr.predict([[4,2,240,20,0,1,15]])

# %%
dataset.hist()

# %%


# %%
import matplotlib.pyplot as plt  
import numpy as np
from sklearn import metrics
y_predict = np.array([4,2,240,20,0,1,15])
fpr, tpr, thresholds = metrics.roc_curve(y, y_predict)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='example estimator')
display.plot()  
plt.show()      

# %%
plt.figure(figsize = (25,25))
dataset.corr()
sns.heatmap(dataset.corr(), annot = True)

# %%


# %%


# %%
