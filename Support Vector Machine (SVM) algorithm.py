#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn.neural_network as nn

df1 = pd.read_csv(r'C:\Users\nikit\Downloads\SpiralWithCluster.csv',delimiter=',')
df1.describe()


# In[2]:


target_variables1=df1.SpectralCluster
predictors1=df1[["x","y"]]

from sklearn.svm import SVC
svc_Model = SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191106, max_iter = -1,probability=True)
thisFit = svc_Model.fit(predictors1, target_variables1) 
y_predictClass = thisFit.predict(predictors1)


# In[3]:


pred_proba_result = pd.DataFrame(data=svc_Model.predict_proba(predictors1),columns = ["P0","P1"])
pred_proba_result["P0"]


# 2b) What is the misclassification rate?

# In[4]:


y_predictClass


# In[5]:


(df1.SpectralCluster).values


# In[6]:


a=0
i=0
j=0
for i in y_predictClass:
    for j in (df1.SpectralCluster).values:
        if i==j:
            a+=1
acc=a/(len(target_variables1))
acc=acc/100
missclassification = 1- acc
missclassification

print("The miscalssification rate is: ",missclassification)


# In[7]:


svc_Model.predict_proba(predictors1)


# In[8]:


svc_Model = SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191106, max_iter = -1,probability=True)
thisFit = svc_Model.fit(predictors1, target_variables1) 

df1["SVMClusterRepresentation"] = svc_Model.predict(predictors1)


# c)	(5 points) Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the points using the predicted SpectralCluster (0 = Red and 1 = Blue).  Besides, plot the hyperplane as a dotted line to the graph.  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.

# In[9]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
wt = svc_Model.coef_[0]
a = -wt[0] / wt[1]
xx = np.linspace(-5, 5)
yy = a * xx - (svc_Model.intercept_[0]) / wt[1]

# plot the line, the points, and the nearest vectors to the plane
carray=['red','blue']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

plt.plot(xx, yy, 'k--')


for i in range(2):
    data = df1[df1["SVMClusterRepresentation"]==i]
    plt.scatter(data.x,data.y,label = (i),c = carray[i])

plt.title("Scatterplot reprenting to Cluster Values")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()


# a)What is the equation of the separating hyperplane?  Please state the coefficients up to seven decimal places.

# In[10]:


print ('THe equation of the seperating hyperplane is')
print (svc_Model.intercept_[0], " + (", wt[0], ") X +(" ,wt[1],") Y = 0")


# In[11]:


def customArcTan (z):
    theta = np.where(z < 0.0, 2.0*np.pi+z, z)
    return (theta)


# In[12]:


trainData = pd.DataFrame(columns = ["radius","theta"])
trainData['radius'] = np.sqrt(df1['x']**2 + df1['y']**2)
trainData['theta'] = (np.arctan2(df1['y'], df1['x'])).apply(customArcTan)


# d)	(10 points) Please express the data as polar coordinates.  Please plot the theta-coordinate against the radius-coordinate in a scatterplot.  Please color-code the points using the SpectralCluster variable (0 = Red and 1 = Blue).  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.

# In[13]:


trainData['class']=df1["SpectralCluster"]
trainData.head()


# In[14]:


colur = ['red','blue']
for i in range(2):
    subdata = trainData[trainData["class"]==i]
    plt.scatter(subdata.radius,subdata.theta,label = (i),c = carray[i])
    
plt.title("Scatterplot-Polar Co-ordinates")
plt.xlabel("Radius")
plt.ylabel('Theta Co-ordinate')
plt.grid()
plt.legend()


# e)	(10 points) You should expect to see three distinct strips of points and a lone point.  Since the SpectralCluster variable has two values, you will create another variable, named Group, and use it as the new target variable. The Group variable will have four values. Value 0 for the lone point on the upper left corner of the chart in (d), values 1, 2,and 3 for the next three strips of points.
# 
# Please plot the theta-coordinate against the radius-coordinate in a scatterplot.  Please color-code the points using the new Group target variable (0 = Red, 1 = Blue, 2 = Green, 3 = Black).  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.
# 

# In[15]:


x = trainData["radius"]
y = trainData['theta'].apply(customArcTan)
svm_dataframe = pd.DataFrame(columns = ['Radius','Theta'])
svm_dataframe['Radius'] = x
svm_dataframe['Theta'] = y

group = []

for grp in range(len(x)):
    if x[grp] < 1.5 and y[grp]>6:
        group.append(0)
     
    elif 2.75 > x[grp]>2.5 and y[grp]>5:
        group.append(1)
    
    elif x[grp] < 2.5 and y[grp]>3 :
        group.append(1)
    
   
        
    elif 2.5<x[grp]<3 and 2<y[grp]<4:
        group.append(2)      
        
    elif x[grp]> 2.5 and y[grp]<3.1:
        group.append(3)
        
    elif x[grp] < 4:
        group.append(2)
        

svm_dataframe['Class'] = group
colors = ['red','blue','green','black']
for i in range(4):
    sub = svm_dataframe[svm_dataframe.Class == i]
    plt.scatter(sub.Radius,sub.Theta,c = colors[i],label=i)
plt.grid()
plt.title("Scatterplot with four Groups")
plt.xlabel("Radius")
plt.ylabel('Theta Co-ordinate')
plt.legend()


# f)	(10 points) Since the graph in (e) has four clearly separable and neighboring segments, we will apply the Support Vector Machine algorithm in a different way.  Instead of applying SVM once on a multi-class target variable, you will SVM three times, each on a binary target variable.
# SVM 0: Group 0 versus Group 1
# SVM 1: Group 1 versus Group 2
# SVM 2: Group 2 versus Group 3
# Please give the equations of the three hyperplanes.
# 

# In[16]:


#SVM to classify class 0 and class 1
svm_1 = SVC(kernel = "linear", random_state=20191108, decision_function_shape='ovr',max_iter=-1,probability = True)
x = svm_dataframe[svm_dataframe['Class'] == 0]
x = x.append(svm_dataframe[svm_dataframe['Class'] == 1])
td = x[['Radius','Theta']]
svm_1.fit(td,x.Class)

w = svm_1.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(1, 2)
yy = a * xx - (svm_1.intercept_[0])/w[1] 

print ('THe equation of the hypercurve for SVM 0 is')
print (svm_1.intercept_[0], " + (", w[0], ") X +(" ,w[1],") Y = 0")

h0_xx = xx * np.cos(yy[:])
h0_yy = xx * np.sin(yy[:])

carray=['red','blue','green','black']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

#Plot ther hyperplane
plt.plot(xx, yy, 'k--')

#SVM to classify class 1 and class 2
svm_1 = SVC(kernel = "linear", random_state=20191108, decision_function_shape='ovr',max_iter=-1,probability = True)
x = svm_dataframe[svm_dataframe['Class'] == 1]
x = x.append(svm_dataframe[svm_dataframe['Class'] == 2])
td = x[['Radius','Theta']]
svm_1.fit(td,x.Class)

w = svm_1.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(1, 4)
yy = a * xx - (svm_1.intercept_[0])/w[1] 
print ('THe equation of the hypercurve for SVM 1 is')
print (svm_1.intercept_[0], " + (", w[0], ") X +(" ,w[1],") Y = 0")

h1_xx = xx * np.cos(yy[:])
h1_yy = xx * np.sin(yy[:])


#Plot ther hyperplane
plt.plot(xx, yy, 'k--')

#SVM to. classify class 2 and class 3
svm_1 = SVC(kernel = "linear", random_state=20191108, decision_function_shape='ovr',max_iter=-1,probability = True)
x = svm_dataframe[svm_dataframe['Class'] == 2]
x = x.append(svm_dataframe[svm_dataframe['Class'] == 3])
td = x[['Radius','Theta']]
svm_1.fit(td,x.Class)

w = svm_1.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(2, 4.5)
yy = a * xx - (svm_1.intercept_[0])/w[1] 
print ('THe equation of the hypercurve for SVM 2 is')
print (svm_1.intercept_[0], " + (", w[0], ") X +(" ,w[1],") Y = 0")

h2_xx = xx * np.cos(yy[:])
h2_yy = xx * np.sin(yy[:])


#Plot ther hyperplane
plt.plot(xx, yy, 'k--')


for i in range(4):
    sub = svm_dataframe[svm_dataframe.Class == i]
    plt.scatter(sub.Radius,sub.Theta,c = carray[i],label=i)
plt.xlabel("Radius")
plt.ylabel("Theta Co-Ordinate")
plt.title("Scatterplot of the polar co-ordinates with 4 diffrent classes seperated by 3 hyperplanes")
plt.legend()


# g)	(5 points) Please plot the theta-coordinate against the radius-coordinate in a scatterplot.  Please color-code the points using the new Group target variable (0 = Red, 1 = Blue, 2 = Green, 3 = Black). Please add the hyperplanes to the graph. To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.

# In[17]:


carray=['red','blue']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

plt.plot(h0_xx, h0_yy, 'k--')
plt.plot(h1_xx, h1_yy, 'k--')
plt.plot(h2_xx, h2_yy, 'k--')

for i in range(2):
    subdata = df1[df1["SpectralCluster"]==i]
    plt.scatter(subdata.x,subdata.y,label = (i),c = carray[i])

plt.title("Scatterplot of the cartesian co-ordinates seperated by the hypercurve ")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")
plt.legend()


# In[18]:


carray=['red','blue']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

#plt.plot(h0_xx, h0_yy, 'k--')
plt.plot(h1_xx, h1_yy, 'k--')
plt.plot(h2_xx, h2_yy, 'k--')

for i in range(2):
    subdata = df1[df1["SpectralCluster"]==i]
    plt.scatter(subdata.x,subdata.y,label = (i),c = carray[i])

plt.title("Scatterplot of the cartesian co-ordinates seperated by the hypercurve (one hypercurve removed)")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




