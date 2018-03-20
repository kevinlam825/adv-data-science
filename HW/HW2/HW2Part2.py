
# coding: utf-8

# # CS5661 Homework 2
# Kevin Lam (CIN: 303061725) 

# ### Importing necessary libraries

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


# In[2]:

#To allow images to be shown inside Jupyter
get_ipython().magic('matplotlib inline')


# ### Loading the dataset and creating a helper function

# In[3]:

#Load dataset
img_df = pd.read_csv('label.csv')

# Simple helper function for retrieving file/names
def get_file_path(name_of_file):
    x = 'Digit/' + str(name_of_file) + '.jpg'
    return x


# ### Create list for column names
# Format will be 'Pixel_XY' with X & Y ranging from 1-8 to represent each pixel location in a 8x8 grid

# In[4]:

column_list = []
for i in range(1,9):
    for j in range(1,9):
        xy = (i*10)+j
        name = 'Pixel_' + str(xy)
        column_list.append(name)

pixel_column_list = column_list.copy() # Keep copy of just the pixel columns for easier referencing in training
column_list.append("Digit")
column_list.append("FileName")

# Create dataframe based on column list
df = pd.DataFrame(columns = [column_list])

df.head()


# ### Iterate through the csv, read and append data to dataframe

# In[5]:

df = df[0:0] # Clear dataframe

for index, row in img_df.iterrows():                    # Loop through the csv
    filename = get_file_path(row['name of the file'])    # Get file path for image
    img = mpimg.imread(filename).reshape(-1)             # Read image and collapse into a 1-dimensional array (8x8 -> 1x64)
    data = list(img)
    data.append(row['digit'])         # Include the digit
    data.append(filename)             # and file path
    s = pd.Series(data, index=column_list)
    df = df.append(s, ignore_index=True) # Append to dataframe

df.head()


# ### Part B & C - Define feature set and target, and train_test_split

# In[6]:

X = df[pixel_column_list]
y = df['Digit']
y = y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

X_train.head()


# ### Part D - Design and Train ANN

# In[7]:

# 1 Hidden Layer with 80 neurons:
my_ANN = MLPClassifier(hidden_layer_sizes=(80,),activation='logistic', 
                       solver='adam', alpha=1e-5, random_state=1, learning_rate_init = 0.002)


my_ANN.fit(X_train, y_train)


# ### Testing ANN on Training Set | Accuracy & Confusion Matrix

# In[8]:

y_predict = my_ANN.predict(X_test)

score = accuracy_score(y_test, y_predict)
print("Accuracy score:",score)


# In[9]:

confusion_matrix(y_test, y_predict)


# ### Part E - Using GridSearchCV

# In[10]:

# define a range for the "number of neurons" in the hidden layer for a network with 1 hidden layer:
neuron_number = [(i,) for i in range(50,200)]

# create a dictionary for grid parameter:
param_grid = dict(hidden_layer_sizes = neuron_number)
#print(param_grid,'\n')

# instantiate the model:
my_ANN = MLPClassifier(activation='logistic', solver='adam', 
                                         alpha=1e-5, random_state=1, 
                                           learning_rate_init = 0.002)

# create the grid, and define the metric for evaluating the model: 
grid = GridSearchCV(my_ANN, param_grid, cv=10, scoring='accuracy')

# fit the grid (start the grid search):
grid.fit(X, y)


# #### GridSearch -> Best Accuracy & Number of Neurons

# In[13]:

print("Best Accuracy:",grid.best_score_)
print("Best number of neurons:",grid.best_params_['hidden_layer_sizes'][0])

