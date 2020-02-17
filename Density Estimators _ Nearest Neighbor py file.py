#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
#describing data
data = pd.read_csv(r'D:\Course_Work\ML\week 2\NormalSample (1).csv')
print(data.head())
a=data.describe()

#1(a)(b)(c)calculating min max q1 q2 IQR binwidth
Q1=np.percentile(data.x,25)
Q3=np.percentile(data.x,75)
IQR1=Q3-Q1
N = len(data)
a=N**(-1/3)
h=2*IQR1*a
bin_width=round(h,2)
minimum_value= min(data.x)
maximum_value= max(data.x)
a=round(minimum_value)
b=round(maximum_value+1)
print(a,b)

#1(a)(b)(c)Printing min max q1 q2 IQR binwidth
print("")
print("Printing min max q1 q2 IQR binwidth")
print("1st Quantile: ",Q1)
print("3rd Quantile: ",Q3)
print("Inter Quartile Range:",IQR1)
print("Total number of observations (N): ",N)
print("binwidth:",bin_width)
print("minimum value:",minimum_value)
print("maximum value:",maximum_value)
print("Largest Integer less than minimum value=a=:",round(minimum_value))
print("Smallest Integer greater than maximum value=b=:",round(maximum_value)+1)

#calculating list of midpoints
h=0.1
midpoints=[]
mid = h/2
m1= a+mid
print(m1) 

Bin_no = round((b-a)/h)
print(Bin_no)
for x in range(Bin_no+1):
    if m1<36:
        midpoints.append(m1)
        m1=m1+h
print(midpoints)

#print(data.x)
#Storing data in Series 
series_ = (data['x'])
#print(series_)
#type(series_)

print("")
#1(d)Probability density estimator and histogram for h=0.1
Probability_density=[]
for m in midpoints:
    count = 0
    h=0.1
    u = (series_- m)/h
    for j in range(len(u)):
        if u[j]<=0.5 and u[j]>-0.5:
            count+=1
    Prob_density=count/(len(u)*h)
    Probability_density.append(Prob_density)
    count = 0

plt.figure(figsize=(6,4))

plt.step(midpoints, Probability_density, where = 'mid', label = 'h = 0.1')    
plt.legend()
plt.grid(True)
plt.show() 
df1 = pd.DataFrame(list(zip(midpoints,Probability_density)), columns =['midpoints','Probability_density'])
print(df1)


print("")
# 1(d) Probability density estimator and histogram for h=0.5
h=0.5
midpoints=[]
mid = h/2
m1= a+mid
print(m1) 

Bin_no = round((b-a)/h)
print(Bin_no)
for x in range(Bin_no+1):
    if m1<36:
        midpoints.append(m1)
        m1=m1+h
Probability_density=[]
for m in midpoints:
    count = 0
    u = (series_- m)/h
    for j in range(len(u)):
        if u[j]<=0.5 and u[j]>-0.5:
            count+=1
    Prob_density=count/(len(u)*h)
    Probability_density.append(Prob_density)
    count = 0

plt.figure(figsize=(6,4))

plt.step(midpoints, Probability_density, where = 'mid', label = 'h = 0.5')    
plt.legend()
plt.grid(True)
plt.show() 

df1 = pd.DataFrame(list(zip(midpoints,Probability_density)), columns =['midpoints','Probability_density'])
print(df1)

print("")
# 1(d) Probability density estimator and histogram for h=1

h=1
midpoints=[]
mid = h/2
m1= a+mid
print(m1) 

Bin_no = round((b-a)/h)
print(Bin_no)
for x in range(Bin_no+1):
    if m1<36:
        midpoints.append(m1)
        m1=m1+h
Probability_density=[]
for m in midpoints:
    count = 0
    u = (series_- m)/h
    for j in range(len(u)):
        if u[j]<=0.5 and u[j]>-0.5:
            count+=1
    Prob_density=count/(len(u)*h)
    Probability_density.append(Prob_density)
    count = 0

plt.figure(figsize=(6,4))

plt.step(midpoints, Probability_density, where = 'mid', label = 'h = 1')    
plt.legend()
plt.grid(True)
plt.show() 

df1 = pd.DataFrame(list(zip(midpoints,Probability_density)), columns =['midpoints','Probability_density'])
print(df1)


print("")
# 1(d) Probability density estimator and histogram for h=2
h=2
midpoints=[]
mid = h/2
m1= a+mid
print(m1) 

Bin_no = round((b-a)/h)
print(Bin_no)
for x in range(Bin_no+1):
    if m1<36:
        midpoints.append(m1)
        m1=m1+h
Probability_density=[]
for m in midpoints:
    count = 0
    u = (series_- m)/h
    for j in range(len(u)):
        if u[j]<=0.5 and u[j]>-0.5:
            count+=1
    Prob_density=count/(len(u)*h)
    Probability_density.append(Prob_density)
    count = 0

plt.figure(figsize=(6,4))

plt.step(midpoints, Probability_density, where = 'mid', label = 'h = 2')    
plt.legend()
plt.grid(True)
plt.show() 

df1 = pd.DataFrame(list(zip(midpoints,Probability_density)), columns =['midpoints','Probability_density'])
print(df1)


# In[2]:



#describing data for Question 2(a)
data_Q2 = pd.read_csv(r'D:\Course_Work\ML\week 2\NormalSample (1).csv',delimiter=',', usecols=["x"])
data_Q2.describe()
#finding Whisker1 and whisker2 for Question2(a)
Q1=np.percentile(data_Q2,25)
Q3=np.percentile(data_Q2,75)
IQR=Q3-Q1
Whisker1_1=Q1-1.5*IQR 
Whisker2_1=Q3+1.5*IQR
print("Results for Question 2(a)")
print("Lower Whisker",Whisker1_1)
print("Upper Whisker",Whisker2_1)
print("")
#describing data for Question 2(b)
data1 = pd.read_csv(r'D:\Course_Work\ML\week 2\NormalSample (1).csv',delimiter=',')
df=data1.groupby('group').describe()
#finding Whisker1 and whisker2 for Question2(b)
q1 = np.percentile(data1[data1.group == 0].x, 25)
q3 = np.percentile(data1[data1.group == 0].x, 75) 
IQR = q3-q1
Whisker1_2=q1-1.5*IQR 
Whisker2_2=q3+1.5*IQR
print("Results for Question 2(b) group 0")
print("Lower Whisker",Whisker1_2)
print("Upper Whisker",Whisker2_2)
print("")
q1 = np.percentile(data1[data1.group == 1].x, 25)
q3 = np.percentile(data1[data1.group == 1].x, 75) 
IQR = q3-q1
Whisker1_3=q1-1.5*IQR 
Whisker2_3=q3+1.5*IQR 
print("Results for Question 2(b) group 1")
print("Lower Whisker",Whisker1_3)
print("Upper Whisker",Whisker2_3)

#plotting box plot Question 2c
print("plotting box plot Question 2c")
boxplot = data.boxplot(column='x',vert=False)

#question 2d
print("plotting box plot Question 2d combined")
x = data_Q2.x
data_grp_0= data1[data1.group==0].x
data_grp_1= data1[data1.group==1].x
Concatination_=pd.concat([x,data_grp_0,data_grp_1],axis=1, keys=['x','data_grp_0','data_grp_1'])
plt.figure(figsize=(10,10))
sns.boxplot(data=Concatination_)

#Outliers in 2d
print("outliers of x for entire data are as follows")
xl = list(x)
for i in xl:
    if i < Whisker1_1:
        print("Outliers less than 1st Whisker are:")
        print(i)
    elif i>Whisker2_1:
        print("Outliers greater than 2st Whisker are:")
        print(i)
        
#Outliers in 2d
print("outliers of x for each data are as follows:")
xl1 = list(data_grp_0)
xl2 = list(data_grp_1)

print("outliers of x for group 0 are as follows:")
for i in xl1:
    if i < Whisker1_2:
        print("Outliers less than 1st Whisker are:")
        print(i)
    elif i>Whisker2_2:
        print("Outliers greater than 2st Whisker are:")
        print(i)
print("outliers of x for group 1 are as follows:")
for i in xl2:
    if i < Whisker1_3:
        print("Outliers less than 1st Whisker are:")
        print(i)
    elif i>Whisker2_3:
        print("Outliers greater than 2st Whisker are:")
        print(i)


# In[3]:



datax = pd.read_csv(r'D:\Course_Work\ML\week 2\Fraud (1).csv',delimiter=',')
datax.head()

#3a Percentage of fraudulent investigation
fraud_check = datax.groupby('FRAUD').describe()
print(fraud_check)
fraud_done=fraud_check.iloc[1,0]
Fraud_not_done=fraud_check.iloc[0,0]
percentage_fraud= (fraud_done/(fraud_done+Fraud_not_done))*100
print("%fraud(rounded of to 4 digits)",round(percentage_fraud,4))

#3b For each interval variable, one box-plot for the fraudulent observations
datax.boxplot(column='TOTAL_SPEND', by='FRAUD', vert=False)
plt.xlabel("TOTAL_SPEND")
plt.ylabel("FRAUD")
plt.show()

datax.boxplot(column='DOCTOR_VISITS', by='FRAUD', vert=False)
plt.xlabel("DOCTOR_VISITS")
plt.ylabel("FRAUD")
plt.show()

datax.boxplot(column='NUM_CLAIMS', by='FRAUD', vert=False)
plt.xlabel("NUM_CLAIMS")
plt.ylabel("FRAUD")
plt.show()

datax.boxplot(column='MEMBER_DURATION', by='FRAUD', vert=False)
plt.xlabel("MEMBER_DURATION")
plt.ylabel("FRAUD")
plt.show()

datax.boxplot(column='OPTOM_PRESC', by='FRAUD', vert=False)
plt.xlabel("OPTOM_PRESC")
plt.ylabel("FRAUD")
plt.show()

datax.boxplot(column='NUM_MEMBERS', by='FRAUD', vert=False)
plt.xlabel("NUM_MEMBERS")
plt.ylabel("FRAUD")
plt.show()

#3(c)
#Orthonormalize interval variables and use the resulting variables for the nearest neighbor analysis. Use only the dimensions whose corresponding eigenvalues are greater than one.
data_frame = pd.read_csv(r'D:\Course_Work\ML\week 2\Fraud (1).csv',delimiter=',',usecols=["TOTAL_SPEND","DOCTOR_VISITS","NUM_CLAIMS","MEMBER_DURATION","OPTOM_PRESC","NUM_MEMBERS"])
print(data_frame)

# Input the matrix X
x = np.matrix(data_frame)

#taking transpose
xtx = x.transpose() * x
print("t(x) * x = \n", xtx)

#Eigenvalue decomposition
evals, evecs = LA.eigh(xtx)
print("Eigenvalues of x = \n", evals)

# Here is the transformation matrix
transf = evecs * LA.inv(np.sqrt(np.diagflat(evals)));
print("Transformation Matrix = \n", transf)
# Here is the transformed X
transf_x = x * transf;
print("The Transformed x = \n", transf_x)

# Check columns of transformed X
xtx = transf_x.transpose() * transf_x;
print("Expect an Identity Matrix = \n", xtx)
type(xtx)

#3(d)
#perform classification
data_x1 =pd.DataFrame(transf_x)
#Fraud_Index = data_.set_index("CASE_ID")
trainData = data_x1
target = datax['FRAUD']
neigh = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(trainData, target)
score_result = nbrs.score(trainData, target)

print("Score result:",score_result)

#3e
#finding neighbors for test data
focal = [[7500, 15, 3, 127, 2, 2]]
transf_focal = focal * transf;
myNeighbors_t = nbrs.kneighbors(transf_focal, return_distance = False)
print("My Neighbors = \n", myNeighbors_t)
print(datax.iloc[list(myNeighbors_t[0])])


# In[ ]:




