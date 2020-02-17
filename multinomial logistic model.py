#!/usr/bin/env python
# coding: utf-8

# #  Assignment 4 : Machine Learning Question 1

# In[1]:


import numpy
import pandas
import scipy
import sympy 
import math

import statsmodels.api as stats

# A function that returns the columnwise product of two dataframes (must have same number of rows)
def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pandas.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y, debug = 'N'):
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(fullX.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)

    # Extract only the non-redundant columns for modeling
    X = fullX.iloc[:, list(inds)]

    # The number of free parameters
    thisDF = len(inds) * (nYCat - 1)

    # Build a multionomial logistic model
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    if (debug == 'Y'):
        print(thisFit.summary())
        print("Model Parameter Estimates:\n", thisParameter)
        print("Model Log-Likelihood Value =", thisLLK)
        print("Number of Free Parameters =", thisDF)

    # Recreat the estimates of the full parameters
    workParams = pandas.DataFrame(numpy.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pandas.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)

hmeq = pandas.read_csv('Purchase_Likelihood.csv',delimiter=',', usecols = ['A', 'group_size', 'homeowner','married_couple'])

hmeq = hmeq.dropna()

# Specify Origin as a categorical variable
y = hmeq['A'].astype('category')

# Specify Group_size, Homeowner and married_couple as categorical variables
xG = pandas.get_dummies(hmeq[['group_size']].astype('category'))
xH = pandas.get_dummies(hmeq[['homeowner']].astype('category'))
xM = pandas.get_dummies(hmeq[['married_couple']].astype('category'))

# Intercept only model
designX = pandas.DataFrame(y.where(y.isnull(), 1))
LLK0, DF0, fullParams0 = build_mnlogit (designX, y, debug = 'Y')


# In[2]:


# Intercept + Group_size
designX = stats.add_constant(xG, prepend=True)
LLK_1G, DF_1G, fullParams_1G = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1G - LLK0)
testDF = DF_1G - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print('  Feature Importance = ', -math.log10(testPValue))


# In[3]:


# Intercept + Group_size + homeowner
designX = xH
designX = designX.join(xG)
designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H, DF_1G_1H, fullParams_1G_1H = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1G_1H - LLK_1G)
testDF = DF_1G_1H - DF_1G
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
#print('  Feature Importance = ', -math.log10(testPValue))


# ### a)	(5 points) List the aliased parameters that you found in your model.
# 
# ### c)	(10 points) After entering a model effect, calculate the Deviance test statistic, its degrees of freedom, and its significance value between the current model and the previous model.  List your Deviance test results by the model effects in a table.

# In[4]:


# Intercept + Group_size + homeowner + married_couple
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)
designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H_1M, DF_1G_1H_1M, fullParams_1G_1H_1M = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1G_1H_1M - LLK_1G_1H)
testDF = DF_1G_1H_1M - DF_1G_1H
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
#print('  Feature Importance = ', -math.log10(testPValue))

# Intercept + Group_size + homeowner + married_couple + Group_size*homeowner
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)
xGH= create_interaction(xG, xH)
designX = designX.join(xGH)
designX = stats.add_constant(designX, prepend=True)
LLK_2GH, DF_2GH, fullParams_2GH = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_2GH - LLK_1G_1H_1M)
testDF = DF_2GH - DF_1G_1H_1M
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
#print('  Feature Importance = ', -math.log10(testPValue))

# Intercept + Group_size + homeowner + married_couple + Group_size*homeowner + homeowner*married_couple
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)
xGH = create_interaction(xG, xH)
xHM = create_interaction(xH, xM)
designX = designX.join(xGH)
designX = designX.join(xHM)
designX = stats.add_constant(designX, prepend=True)
LLK_2HM, DF_2HM, fullParams_2HM = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_2HM - LLK_2GH)
testDF = DF_2HM - DF_2GH
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
#print('  Feature Importance = ', -math.log10(testPValue))


# In[5]:


n = [0, 1, 2, 3, 5, 7, 9, 11, 13, 17]
m = list(range(len(designX.columns)))
#designX.columns[n]
print("Aliased parameters in the model")
for i in m:
    if i not in n:
        print(designX.columns[i])


# ### b)	(5 points) How many degrees of freedom do you have in your model?

# In[6]:


print(DF_2HM)


# In[7]:


Final = []
GS = [1,2,3,4]
H = [0,1]
M = [0,1]

for i in GS:
    for j in H:
        for k in M:
            Final.append([i,j,k])

df = pandas.DataFrame(Final, columns=['group_size','homeowner','married_couple'])

df_groupsize = pandas.get_dummies(df[['group_size']].astype('category'))
FinalX = df_groupsize

df_homeowner = pandas.get_dummies(df[['homeowner']].astype('category'))
FinalX = FinalX.join(df_homeowner)

df_marriedcouple = pandas.get_dummies(df[['married_couple']].astype('category'))
FinalX = FinalX.join(df_marriedcouple)

df_groupsize_h = create_interaction(df_groupsize, df_homeowner)
df_groupsize_h = pandas.get_dummies(df_groupsize_h)
FinalX = FinalX.join(df_groupsize_h)

df_homeowner_m = create_interaction(df_homeowner, df_marriedcouple)
df_homeowner_m = pandas.get_dummies(df_homeowner_m)
FinalX = FinalX.join(df_homeowner_m)


FinalX = stats.add_constant(FinalX, prepend=True)


# ### d)	(5 points) Calculate the Feature Importance Index as the negative base-10 logarithm of the significance value.  List your indices by the model effects.

# In[8]:


FeatureImportance = [4.347870389027117e-210,4.306457217534288e-19,5.512105969198056e-52,4.13804354648637e-16 ]
for i in FeatureImportance:
    print(-math.log10(i))


# ### e)	(10 points) For each of the sixteen possible value combinations of the three features, calculate the predicted probabilities for A = 0, 1, 2 based on the multinomial logistic model.  List your answers in a table with proper labelling.

# In[9]:


logit = stats.MNLogit(y, designX)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)           
PP = thisFit.predict(FinalX)
print(PP)


# In[11]:


import pandas as pd
df1=pd.DataFrame(PP)
df2=pd.DataFrame(Final)
result= pd.merge(df2,df1, left_index=True,right_index=True)
result
result1=result.rename(columns={'0_x':'Group_size','1_x':'Homeowner','2_x':'married_couple','0_y':'A=0','1_y':'A=1','2_y':'A=2'})
result1


# ### f)	(5 points) Based on your model, what values of group_size, homeowner, and married_couple will maximize the odds value Prob(A=1) / Prob(A = 0)?  What is that maximum odd value?
# 

# In[ ]:


(PP[1]/PP[0])


# In[ ]:


Final[3]


# ### g)	(5 points) Based on your model, what is the odds ratio for group_size = 3 versus group_size = 1, and A = 2 versus A = 0?  Mathematically, the odds ratio is (Prob(A=2)/Prob(A=0) | group_size = 3) / ((Prob(A=2)/Prob(A=0) | group_size = 1).

# In[ ]:


print(fullParams_2HM)


# Taking A=0 as reference target category
# 
# Loge((Prob(A=2)/Prob(A=0) | group_size = 3) ) - loge((Prob(A=2)/Prob(A=0) | group_size = 1))
# = Parameter of (group_size = 3 | A=2) – Parameter of (group_size = 1 | A=2)
#               = 0.527471 - 0.801493
#               = -0.274022
#               Taking exponent of the previous value: exp(-0.274022) = 0.76031534813
# 

# ### h)	(5 points) Based on your model, what is the odds ratio for homeowner = 1 versus homeowner = 0, and A = 0 versus A = 1?  Mathematically, the odds ratio is (Prob(A=0)/Prob(A=1) | homeowner = 1) / ((Prob(A=0)/Prob(A=1) | homeowner = 0).

# Log (Prob(A=0)/Prob(A=1) | homeowner = 1) - log((Prob(A=0)/Prob(A=1) | homeowner = 0)
# = (0.800157 – 1.505554 * g1 – 1.164638 * g2 – 0.654639 * g3 + 0.212483 (1-m)
# 
# Exp (Prob(A=0)/Prob(A=1) | homeowner = 1) - log((Prob(A=0)/Prob(A=1) | homeowner = 0)

# In[ ]:




