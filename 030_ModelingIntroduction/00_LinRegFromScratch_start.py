#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns
import matplotlib.pyplot as plt

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
X = torch.from_numpy(np.array(cars['wt'], dtype=np.float32).reshape(-1,1))
Y = torch.from_numpy(np.array(cars['mpg'], dtype=np.float32))

W = torch.rand(1, requires_grad=True, dtype = torch.float32)
b = torch.zeros(1, requires_grad=True, dtype = torch.float32)

num_epochs = 1000
lr = 0.001

list_loss = []
for epoch in range(num_epochs):
    for i in range(len(X)):
        y_pred = W @ X[i] + b;
        loss = torch.pow((y_pred - Y[i]), 2)
        loss_val = loss.data[0]
        
        loss.backward()

        with torch.no_grad():
            W -= lr * W.grad
            b -= lr * b.grad
            W.grad.zero_()
            b.grad.zero_()
    
    #print ("Loss: ", loss_val)
    list_loss.append(float(loss_val))

print (list_loss)
sns.lineplot(data=list_loss)
plt.show()



#%% training

#%% check results
# %%
y_pred = X * W + b;
X_list = cars.wt.values
sns.scatterplot(y=y_pred.detach().numpy().reshape(-1), x=X_list)

# %% (Statistical) Linear Regression


# %% create graph visualisation
# make sure GraphViz is installed (https://graphviz.org/download/)
# if not computer restarted, append directly to PATH variable
import os
from torchviz import make_dot
#os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin'
make_dot(loss)
# %%
