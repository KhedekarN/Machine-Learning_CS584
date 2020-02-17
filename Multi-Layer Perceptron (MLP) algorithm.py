#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df1 = pd.read_csv(r'C:\Users\nikit\Downloads\SpiralWithCluster.csv',delimiter=',')


# In[2]:


df1.head()


# 1 a) What percent of the observations have SpectralCluster equals to 1?

# In[3]:


crossTable1 = pd.crosstab(index = df1.SpectralCluster, columns = ["Counts"], margins = True, dropna=True)
crossTable1['Percentage']= 100*(crossTable1['Counts']/len(df1))
crossTable1= crossTable1.drop(columns = ['All'])
crossTable1


# taking target and predictors out from dataframe

# In[4]:


target_variable=df1.SpectralCluster
predictors=df1[["x","y"]]


# In[5]:


import sklearn.neural_network as nn
def Build_NN(activation_function, nhiddenlayers, nneurons):
    clf1 = nn.MLPClassifier(activation = activation_function,
                             hidden_layer_sizes = (nneurons,)*nhiddenlayers,
                             learning_rate_init=0.1,
                             max_iter=5000, random_state = 20191108,
                             solver = "lbfgs")
    thisFit=clf1.fit(predictors,target_variable)
    target= clf1.predict(predictors)
    
    Loss = clf1.loss_
    predicted_probability = pd.DataFrame(data=clf1.predict_proba(predictors),columns = ["P0","P1"])
    misclassification = misclassification_rate(predicted_probability["P1"],target_variable)
    
    return (Loss,misclassification,clf1.n_iter_)


# In[8]:


def misclassification_rate(predicted_probability,target_variable):
    
    result =[]
    limit = 0.50 #from 1a 
    count = 0
    
    for i in predicted_probability:
        if i <= limit:
            result.append(0)
        else:
            result.append(1)
    for j in range(len(result)):
        if result[j] != target_variable[j]:
            count += 1
    return (count/len(result))  


# 1b) You will search for the neural network that yields the lowest loss value and the lowest misclassification rate.  You will use your answer in (a) as the threshold for classifying an observation into SpectralCluster = 1. Your search will be done over a grid that is formed by cross-combining the following attributes: (1) activation function: identity, logistic, relu, and tanh; (2) number of hidden layers: 1, 2, 3, 4, and 5; and (3) number of neurons: 1 to 10 by 1.  List your optimal neural network for each activation function in a table.  Your table will have four rows, one for each activation function.  Your table will have five columns: (1) activation function, (2) number of layers, (3) number of neurons per layer, (4) number of iterations performed, (5) the loss value, and (6) the misclassification rate.

# In[9]:


import numpy as np
table_data1 = pd.DataFrame(columns = ['Number of Layers', 'Number of Neurons', 'Loss', 'Misclassification Rate', "Activation Function","Number of Iterations"])

activaton_function = ['relu' , 'identity', 'logistic', 'tanh']

for a_fun in activaton_function:
    for nhidden in np.arange(1,6):
        for neurons in np.arange(1,11):
            Loss, rsquared, iterations = Build_NN(activation_function = a_fun, nhiddenlayers = nhidden, nneurons = neurons)
            table_data1 = table_data1.append(pd.DataFrame([[nhidden, neurons, Loss, rsquared , a_fun, iterations]],columns = ['Number of Layers', 'Number of Neurons', 'Loss', 'Misclassification Rate', "Activation Function","Number of Iterations"]))


# In[10]:


table_data1.sort_values(by=['Loss','Misclassification Rate'])
print(table_data1.sort_values(by=['Loss','Misclassification Rate']))


# d)Which activation function, number of layers, and number of neurons per layer give the lowest loss and the lowest misclassification rate?  What are the loss and the misclassification rate?  How many iterations are performed?

# In[11]:


relu=table_data1[table_data1["Activation Function"]=="relu"]
min_relu= relu[relu["Loss"]==relu["Loss"].min()]

identity=table_data1[table_data1["Activation Function"]=="identity"]
min_identity= identity[identity["Loss"]==identity["Loss"].min()]

logistic = table_data1[table_data1["Activation Function"] == "logistic"]
min_logistic = logistic[logistic["Loss"] == logistic["Loss"].min()]

tanh = table_data1[table_data1["Activation Function"] == "tanh"]
min_tanh = tanh[tanh["Loss"] == tanh["Loss"].min()]

min_relu.append([min_identity,min_logistic,min_tanh])


# In[13]:


clf2 = nn.MLPClassifier(activation = "relu", hidden_layer_sizes = (8,)*4, learning_rate_init=0.1,  max_iter=5000, random_state = 20191108, solver = "lbfgs",verbose=True)
model = clf2.fit(predictors,target_variable)
predicted_probability = pd.DataFrame(data=clf2.predict_proba(predictors),columns = ["P0","P1"])
misclassification = misclassification_rate(predicted_probability["P0"],target_variable)
pred = clf2.predict(predictors)
df1['NLPpredictions'] = pred


# e) Please plot the y-coordinate against the x-coordinate in a scatterplot. Please color-code the points using the predicted SpectralCluster (0 = Red and 1 = Blue) from the optimal MLP in (d). To obtain the full credits, you should properly label the axes, the legend, and the chart title. Also, grid lines should be added to the axes.

# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.color_palette("RdBu", n_colors=7)
#fig, ax = plt.subplots(1, 1)
#ax.grid(b=True, which='major')
clr = ['red','blue']
for i in range (len(clr)):
    Data = df1[df1['NLPpredictions']==i]
    plt.scatter(Data.x,Data.y,c = clr[i],label=i)
    plt.legend()
plt.title("Scatterplot for predicted cluster values of neural network")
plt.xlabel("X")
plt.grid()
plt.ylabel("Y")
plt.legend()


# f) What is the count, the mean and the standard deviation of the predicted probability Prob(SpectralCluster = 1) from the optimal MLP in (d) by value of the SpectralCluster?  Please give your answers up to the 10 decimal places.

# In[15]:


list1 = []
for i in predicted_probability["P1"]:
    if i > 0.5:
        list1.append(i)
df3=pd.DataFrame(list1)

print("The mean of the predicted probability Prob(SpectralCluster = 1) ",round(np.mean(list1),10))
print("The standard deviation of the predicted probability Prob(SpectralCluster = 1)",round(np.std(list1),10))

df3.describe()


# c)What is the activation function for the output layer?

# In[17]:


print("The activation function for the output layer is",clf2.out_activation_)


# In[ ]:




