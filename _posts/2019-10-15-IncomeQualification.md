---
title: "Income Qualification"
date: "2019-10-15"
tags: [data science, Classification, Ensemble Learning, K-Fold, Cross Validation, Kaggle Dataset]
header:
excerpt: "Data Science, Classification, Ensemble Learning, K-Fold, Cross Validation, Kaggle Dataset"
mathjax: "true"
---
### Income Qualification
#### DESCRIPTION

Identify the level of income qualification needed for the families in Latin America.

#### Problem Statement Scenario:
Many social programs have a hard time ensuring that the right people are given enough aid. It’s tricky when a program focuses on the poorest segment of the population. This segment of the population can’t provide the necessary income and expense records to prove that they qualify.

In Latin America, a popular method called Proxy Means Test (PMT) uses an algorithm to verify income qualification. With PMT, agencies use a model that considers a family’s observable household attributes like the material of their walls and ceiling or the assets found in their homes to
classify them and predict their level of need.

While this is an improvement, accuracy remains a problem as the region’s population grows and poverty declines.

The Inter-American Development Bank (IDB)believes that new methods beyond traditional econometrics, based on a dataset of Costa Rican household characteristics, might help improve PMT’s performance.

### Analysis Tasks to be performed:
1. Identify the output variable.
2. Understand the type of data.
3. Check if there are any biases in your dataset.
4. Check whether all members of the house have the same poverty level.
5. Check if there is a house without a family head.
6. Set poverty level of the members and the head of the house within a family.
7. Count how many null values are existing in columns.
8. Remove null value rows of the target variable.
9. Predict the accuracy using random forest classifier.
10. Check the accuracy using random forest with cross validation.

#### Core Data fields
1. Id - a unique identifier for each row. 
2. Target - the target is an ordinal variable indicating groups of income levels. <br>
    1 = extreme poverty 
    2 = moderate poverty 
    3 = vulnerable households 
    4 = non vulnerable households 
3. idhogar - this is a unique identifier for each household. This can be used to create household-wide features, etc. All rows in a given household will have a matching value for this identifier.
4. parentesco1 - indicates if this person is the head of the household.

Note: that ONLY the heads of household are used in scoring. All household members are included in train +test data sets, but only heads of households are scored.

## Step 1: Understand the Data

#### 1.1 Import necessary Libraries


```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()


import warnings
warnings.filterwarnings('ignore')
```

#### 1.2 Load Data


```python
df_income_train = pd.read_csv("Dataset/train.csv")
df_income_test =  pd.read_csv("Dataset/test.csv")
```



#### 1.3 Explore Train dataset
View first 5 records of train dataset


```python
df_income_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>v2a1</th>
      <th>hacdor</th>
      <th>rooms</th>
      <th>hacapo</th>
      <th>v14a</th>
      <th>refrig</th>
      <th>v18q</th>
      <th>v18q1</th>
      <th>r4h1</th>
      <th>...</th>
      <th>SQBescolari</th>
      <th>SQBage</th>
      <th>SQBhogar_total</th>
      <th>SQBedjefe</th>
      <th>SQBhogar_nin</th>
      <th>SQBovercrowding</th>
      <th>SQBdependency</th>
      <th>SQBmeaned</th>
      <th>agesq</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID_279628684</td>
      <td>190000.0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>...</td>
      <td>100</td>
      <td>1849</td>
      <td>1</td>
      <td>100</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>1849</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ID_f29eb3ddd</td>
      <td>135000.0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>...</td>
      <td>144</td>
      <td>4489</td>
      <td>1</td>
      <td>144</td>
      <td>0</td>
      <td>1.000000</td>
      <td>64.0</td>
      <td>144.0</td>
      <td>4489</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ID_68de51c94</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>...</td>
      <td>121</td>
      <td>8464</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.250000</td>
      <td>64.0</td>
      <td>121.0</td>
      <td>8464</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ID_d671db89c</td>
      <td>180000.0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>...</td>
      <td>81</td>
      <td>289</td>
      <td>16</td>
      <td>121</td>
      <td>4</td>
      <td>1.777778</td>
      <td>1.0</td>
      <td>121.0</td>
      <td>289</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ID_d56d6f5f5</td>
      <td>180000.0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>...</td>
      <td>121</td>
      <td>1369</td>
      <td>16</td>
      <td>121</td>
      <td>4</td>
      <td>1.777778</td>
      <td>1.0</td>
      <td>121.0</td>
      <td>1369</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 143 columns</p>
</div>




```python
df_income_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9557 entries, 0 to 9556
    Columns: 143 entries, Id to Target
    dtypes: float64(8), int64(130), object(5)
    memory usage: 10.4+ MB
    

#### 1.4 Explore Test dataset
View first 5 records of test dataset


```python
df_income_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>v2a1</th>
      <th>hacdor</th>
      <th>rooms</th>
      <th>hacapo</th>
      <th>v14a</th>
      <th>refrig</th>
      <th>v18q</th>
      <th>v18q1</th>
      <th>r4h1</th>
      <th>...</th>
      <th>age</th>
      <th>SQBescolari</th>
      <th>SQBage</th>
      <th>SQBhogar_total</th>
      <th>SQBedjefe</th>
      <th>SQBhogar_nin</th>
      <th>SQBovercrowding</th>
      <th>SQBdependency</th>
      <th>SQBmeaned</th>
      <th>agesq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID_2f6873615</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>0</td>
      <td>16</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>2.25</td>
      <td>0.25</td>
      <td>272.25</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ID_1c78846d2</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>41</td>
      <td>256</td>
      <td>1681</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>2.25</td>
      <td>0.25</td>
      <td>272.25</td>
      <td>1681</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ID_e5442cf6a</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>41</td>
      <td>289</td>
      <td>1681</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>2.25</td>
      <td>0.25</td>
      <td>272.25</td>
      <td>1681</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ID_a8db26a79</td>
      <td>NaN</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>...</td>
      <td>59</td>
      <td>256</td>
      <td>3481</td>
      <td>1</td>
      <td>256</td>
      <td>0</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>256.00</td>
      <td>3481</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ID_a62966799</td>
      <td>175000.0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>...</td>
      <td>18</td>
      <td>121</td>
      <td>324</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.25</td>
      <td>64.00</td>
      <td>NaN</td>
      <td>324</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 142 columns</p>
</div>




```python
df_income_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 23856 entries, 0 to 23855
    Columns: 142 entries, Id to agesq
    dtypes: float64(8), int64(129), object(5)
    memory usage: 25.8+ MB
    

Looking at the train and test dataset we noticed that the following:   
Train dataset:  
    Rows: 9557 entries, 0 to 9556  
    Columns: 143 entries, Id to Target  
    Column dtypes: float64(8), int64(130), object(5)  
        
Test dataset:  
    Rows: 23856 entries, 0 to 23855  
    Columns: 142 entries, Id to agesq  
    dtypes: float64(8), int64(129), object(5)  
          
The important piece of information here is that we don't have 'Target' feature in Test Dataset. There are 5 object type, 130(Train set)/ 129 (test set) integer type and 8 float type features. Lets look at those features next.


```python
#List the columns for different datatypes:
print('Integer Type: ')
print(df_income_train.select_dtypes(np.int64).columns)
print('\n')
print('Float Type: ')
print(df_income_train.select_dtypes(np.float64).columns)
print('\n')
print('Object Type: ')
print(df_income_train.select_dtypes(np.object).columns)
```

    Integer Type: 
    Index(['hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q', 'r4h1', 'r4h2',
           'r4h3', 'r4m1',
           ...
           'area1', 'area2', 'age', 'SQBescolari', 'SQBage', 'SQBhogar_total',
           'SQBedjefe', 'SQBhogar_nin', 'agesq', 'Target'],
          dtype='object', length=130)
    
    
    Float Type: 
    Index(['v2a1', 'v18q1', 'rez_esc', 'meaneduc', 'overcrowding',
           'SQBovercrowding', 'SQBdependency', 'SQBmeaned'],
          dtype='object')
    
    
    Object Type: 
    Index(['Id', 'idhogar', 'dependency', 'edjefe', 'edjefa'], dtype='object')
    


```python
df_income_train.select_dtypes('int64').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hacdor</th>
      <th>rooms</th>
      <th>hacapo</th>
      <th>v14a</th>
      <th>refrig</th>
      <th>v18q</th>
      <th>r4h1</th>
      <th>r4h2</th>
      <th>r4h3</th>
      <th>r4m1</th>
      <th>...</th>
      <th>area1</th>
      <th>area2</th>
      <th>age</th>
      <th>SQBescolari</th>
      <th>SQBage</th>
      <th>SQBhogar_total</th>
      <th>SQBedjefe</th>
      <th>SQBhogar_nin</th>
      <th>agesq</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>43</td>
      <td>100</td>
      <td>1849</td>
      <td>1</td>
      <td>100</td>
      <td>0</td>
      <td>1849</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>67</td>
      <td>144</td>
      <td>4489</td>
      <td>1</td>
      <td>144</td>
      <td>0</td>
      <td>4489</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>92</td>
      <td>121</td>
      <td>8464</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>8464</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>17</td>
      <td>81</td>
      <td>289</td>
      <td>16</td>
      <td>121</td>
      <td>4</td>
      <td>289</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>37</td>
      <td>121</td>
      <td>1369</td>
      <td>16</td>
      <td>121</td>
      <td>4</td>
      <td>1369</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 130 columns</p>
</div>



#### 1.5 Find columns with null values


```python
#Find columns with null values
null_counts=df_income_train.select_dtypes('int64').isnull().sum()
null_counts[null_counts > 0]
```




    Series([], dtype: int64)




```python
df_income_train.select_dtypes('float64').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v2a1</th>
      <th>v18q1</th>
      <th>rez_esc</th>
      <th>meaneduc</th>
      <th>overcrowding</th>
      <th>SQBovercrowding</th>
      <th>SQBdependency</th>
      <th>SQBmeaned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>190000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>135000.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>64.0</td>
      <td>144.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>0.500000</td>
      <td>0.250000</td>
      <td>64.0</td>
      <td>121.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>180000.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>1.333333</td>
      <td>1.777778</td>
      <td>1.0</td>
      <td>121.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>180000.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>1.333333</td>
      <td>1.777778</td>
      <td>1.0</td>
      <td>121.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Find columns with null values
null_counts=df_income_train.select_dtypes('float64').isnull().sum()
null_counts[null_counts > 0]
```




    v2a1         6860
    v18q1        7342
    rez_esc      7928
    meaneduc        5
    SQBmeaned       5
    dtype: int64




```python
df_income_train.select_dtypes('object').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>idhogar</th>
      <th>dependency</th>
      <th>edjefe</th>
      <th>edjefa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID_279628684</td>
      <td>21eb7fcc1</td>
      <td>no</td>
      <td>10</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ID_f29eb3ddd</td>
      <td>0e5d7a658</td>
      <td>8</td>
      <td>12</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ID_68de51c94</td>
      <td>2c7317ea8</td>
      <td>8</td>
      <td>no</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ID_d671db89c</td>
      <td>2b58d945f</td>
      <td>yes</td>
      <td>11</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ID_d56d6f5f5</td>
      <td>2b58d945f</td>
      <td>yes</td>
      <td>11</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Find columns with null values
null_counts=df_income_train.select_dtypes('object').isnull().sum()
null_counts[null_counts > 0]
```




    Series([], dtype: int64)



Looking at the different types of data and null values for each feature. We found the following:
    1. No null values for Integer type features.
    2. No null values for float type features.
    3. For Object types
        v2a1         6860
        v18q1        7342
        rez_esc      7928
        meaneduc        5
        SQBmeaned       5
        
    We also noticed that object type features dependency, edjefe, edjefa have mixed values.
    Lets fix the data for features with null values and features with mixed values

## Step 3: Data Cleaning 

### 3.1 Lets fix the column with mixed values.    

According to the documentation for these columns:  
dependency: Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)   
edjefe: years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0   
edjefa: years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0   

For these three variables, it seems "yes" = 1 and "no" = 0. We can correct the variables using a mapping and convert to floats.


```python
mapping={'yes':1,'no':0}

for df in [df_income_train, df_income_test]:
    df['dependency'] =df['dependency'].replace(mapping).astype(np.float64)
    df['edjefe'] =df['edjefe'].replace(mapping).astype(np.float64)
    df['edjefa'] =df['edjefa'].replace(mapping).astype(np.float64)
    
df_income_train[['dependency','edjefe','edjefa']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dependency</th>
      <th>edjefe</th>
      <th>edjefa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9557.000000</td>
      <td>9557.000000</td>
      <td>9557.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.149550</td>
      <td>5.096788</td>
      <td>2.896830</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.605993</td>
      <td>5.246513</td>
      <td>4.612056</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.666667</td>
      <td>6.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.333333</td>
      <td>9.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
      <td>21.000000</td>
      <td>21.000000</td>
    </tr>
  </tbody>
</table>
</div>



### 3.2 Lets fix the column with null values   

According to the documentation for these columns:   

   v2a1       (total nulls: 6860) : Monthly rent payment   
   v18q1      (total nulls: 7342) : number of tablets household owns   
   rez_esc    (total nulls: 7928) : Years behind in school   
   meaneduc   (total nulls: 5) : average years of education for adults (18+)   
   SQBmeaned  (total nulls: 5) : square of the mean years of education of adults (>=18) in the household 142  




```python
# 1. Lets look at v2a1 (total nulls: 6860) : Monthly rent payment 
# why the null values, Lets look at few rows with nulls in v2a1
# Columns related to  Monthly rent payment
# tipovivi1, =1 own and fully paid house
# tipovivi2, "=1 own,  paying in installments"
# tipovivi3, =1 rented
# tipovivi4, =1 precarious 
# tipovivi5, "=1 other(assigned,  borrowed)"
```


```python
data = df_income_train[df_income_train['v2a1'].isnull()].head()

columns=['tipovivi1','tipovivi2','tipovivi3','tipovivi4','tipovivi5']
data[columns]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tipovivi1</th>
      <th>tipovivi2</th>
      <th>tipovivi3</th>
      <th>tipovivi4</th>
      <th>tipovivi5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Variables indicating home ownership
own_variables = [x for x in df_income_train if x.startswith('tipo')]

# Plot of the home ownership variables for home missing rent payments
df_income_train.loc[df_income_train['v2a1'].isnull(), own_variables].sum().plot.bar(figsize = (10, 8),
                                                                        color = 'green',
                                                              edgecolor = 'k', linewidth = 2);
plt.xticks([0, 1, 2, 3, 4],
           ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'],
          rotation = 20)
plt.title('Home Ownership Status for Households Missing Rent Payments', size = 18);
```


![png](/images/IncomeQualification_files/IncomeQualification_31_0.png)



```python
#Looking at the above data it makes sense that when the house is fully paid, there will be no monthly rent payment.
#Lets add 0 for all the null values.
for df in [df_income_train, df_income_test]:
    df['v2a1'].fillna(value=0, inplace=True)

df_income_train[['v2a1']].isnull().sum()
   
```




    v2a1    0
    dtype: int64




```python
# 2. Lets look at v18q1 (total nulls: 7342) : number of tablets household owns 
# why the null values, Lets look at few rows with nulls in v18q1
# Columns related to  number of tablets household owns 
# v18q, owns a tablet
```


```python
# Since this is a household variable, it only makes sense to look at it on a household level, 
# so we'll only select the rows for the head of household.

# Heads of household
heads = df_income_train.loc[df_income_train['parentesco1'] == 1].copy()
heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())

```




    v18q
    0    2318
    1       0
    Name: v18q1, dtype: int64




```python
plt.figure(figsize = (8, 6))
col='v18q1'
df_income_train[col].value_counts().sort_index().plot.bar(color = 'blue',
                                             edgecolor = 'k',
                                             linewidth = 2)
plt.xlabel(f'{col}'); plt.title(f'{col} Value Counts'); plt.ylabel('Count')
plt.show();
```


![png](/images/IncomeQualification_files/IncomeQualification_35_0.png)



```python
#Looking at the above data it makes sense that when owns a tablet column is 0, there will be no number of tablets household owns.
#Lets add 0 for all the null values.
for df in [df_income_train, df_income_test]:
    df['v18q1'].fillna(value=0, inplace=True)

df_income_train[['v18q1']].isnull().sum()
```




    v18q1    0
    dtype: int64




```python
# 3. Lets look at rez_esc    (total nulls: 7928) : Years behind in school  
# why the null values, Lets look at few rows with nulls in rez_esc
# Columns related to Years behind in school 
# Age in years

# Lets look at the data with not null values first.
df_income_train[df_income_train['rez_esc'].notnull()]['age'].describe()
```




    count    1629.000000
    mean       12.258441
    std         3.218325
    min         7.000000
    25%         9.000000
    50%        12.000000
    75%        15.000000
    max        17.000000
    Name: age, dtype: float64




```python
#From the above , we see that when min age is 7 and max age is 17 for Years, then the 'behind in school' column has a value.
#Lets confirm
df_income_train.loc[df_income_train['rez_esc'].isnull()]['age'].describe()
```




    count    7928.000000
    mean       38.833249
    std        20.989486
    min         0.000000
    25%        24.000000
    50%        38.000000
    75%        54.000000
    max        97.000000
    Name: age, dtype: float64




```python
df_income_train.loc[(df_income_train['rez_esc'].isnull() & ((df_income_train['age'] > 7) & (df_income_train['age'] < 17)))]['age'].describe()
#There is one value that has Null for the 'behind in school' column with age between 7 and 17 
```




    count     1.0
    mean     10.0
    std       NaN
    min      10.0
    25%      10.0
    50%      10.0
    75%      10.0
    max      10.0
    Name: age, dtype: float64




```python
df_income_train[(df_income_train['age'] ==10) & df_income_train['rez_esc'].isnull()].head()
df_income_train[(df_income_train['Id'] =='ID_f012e4242')].head()
#there is only one member in household for the member with age 10 and who is 'behind in school'. This explains why the member is 
#behind in school.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>v2a1</th>
      <th>hacdor</th>
      <th>rooms</th>
      <th>hacapo</th>
      <th>v14a</th>
      <th>refrig</th>
      <th>v18q</th>
      <th>v18q1</th>
      <th>r4h1</th>
      <th>...</th>
      <th>SQBescolari</th>
      <th>SQBage</th>
      <th>SQBhogar_total</th>
      <th>SQBedjefe</th>
      <th>SQBhogar_nin</th>
      <th>SQBovercrowding</th>
      <th>SQBdependency</th>
      <th>SQBmeaned</th>
      <th>agesq</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2514</th>
      <td>ID_f012e4242</td>
      <td>160000.0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>100</td>
      <td>9</td>
      <td>121</td>
      <td>1</td>
      <td>2.25</td>
      <td>0.25</td>
      <td>182.25</td>
      <td>100</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 143 columns</p>
</div>




```python
#from above we see that  the 'behind in school' column has null values 
# Lets use the above to fix the data
for df in [df_income_train, df_income_test]:
    df['rez_esc'].fillna(value=0, inplace=True)
df_income_train[['rez_esc']].isnull().sum()
```




    rez_esc    0
    dtype: int64




```python
#Lets look at meaneduc   (total nulls: 5) : average years of education for adults (18+)  
# why the null values, Lets look at few rows with nulls in meaneduc
# Columns related to average years of education for adults (18+)  
# edjefe, years of education of male head of household, based on the interaction of escolari (years of education),
#    head of household and gender, yes=1 and no=0
# edjefa, years of education of female head of household, based on the interaction of escolari (years of education), 
#    head of household and gender, yes=1 and no=0 
# instlevel1, =1 no level of education
# instlevel2, =1 incomplete primary 
```


```python
data = df_income_train[df_income_train['meaneduc'].isnull()].head()

columns=['edjefe','edjefa','instlevel1','instlevel2']
data[columns][data[columns]['instlevel1']>0].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>edjefe</th>
      <th>edjefa</th>
      <th>instlevel1</th>
      <th>instlevel2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#from the above, we find that meaneduc is null when no level of education is 0
#Lets fix the data
for df in [df_income_train, df_income_test]:
    df['meaneduc'].fillna(value=0, inplace=True)
df_income_train[['meaneduc']].isnull().sum()
```




    meaneduc    0
    dtype: int64




```python
#Lets look at SQBmeaned  (total nulls: 5) : square of the mean years of education of adults (>=18) in the household 142  
# why the null values, Lets look at few rows with nulls in SQBmeaned
# Columns related to average years of education for adults (18+)  
# edjefe, years of education of male head of household, based on the interaction of escolari (years of education),
#    head of household and gender, yes=1 and no=0
# edjefa, years of education of female head of household, based on the interaction of escolari (years of education), 
#    head of household and gender, yes=1 and no=0 
# instlevel1, =1 no level of education
# instlevel2, =1 incomplete primary 
```


```python
data = df_income_train[df_income_train['SQBmeaned'].isnull()].head()

columns=['edjefe','edjefa','instlevel1','instlevel2']
data[columns][data[columns]['instlevel1']>0].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>edjefe</th>
      <th>edjefa</th>
      <th>instlevel1</th>
      <th>instlevel2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#from the above, we find that SQBmeaned is null when no level of education is 0
#Lets fix the data
for df in [df_income_train, df_income_test]:
    df['SQBmeaned'].fillna(value=0, inplace=True)
df_income_train[['SQBmeaned']].isnull().sum()
```




    SQBmeaned    0
    dtype: int64




```python
#Lets look at the overall data
null_counts = df_income_train.isnull().sum()
null_counts[null_counts > 0].sort_values(ascending=False)
```




    Series([], dtype: int64)



### 3.3 Lets look at the target column

Lets see if records belonging to same household has same target/score.


```python
# Groupby the household and figure out the number of unique values
all_equal = df_income_train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))
```

    There are 85 households where the family members do not all have the same target.
    


```python
#Lets check one household
df_income_train[df_income_train['idhogar'] == not_equal.index[0]][['idhogar', 'parentesco1', 'Target']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>idhogar</th>
      <th>parentesco1</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7651</th>
      <td>0172ab1d9</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7652</th>
      <td>0172ab1d9</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7653</th>
      <td>0172ab1d9</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7654</th>
      <td>0172ab1d9</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7655</th>
      <td>0172ab1d9</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Lets use Target value of the parent record (head of the household) and update rest. But before that lets check
# if all families has a head. 

households_head = df_income_train.groupby('idhogar')['parentesco1'].sum()

# Find households without a head
households_no_head = df_income_train.loc[df_income_train['idhogar'].isin(households_head[households_head == 0].index), :]

print('There are {} households without a head.'.format(households_no_head['idhogar'].nunique()))
```

    There are 15 households without a head.
    


```python
# Find households without a head and where Target value are different
households_no_head_equal = households_no_head.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
print('{} Households with no head have different Target value.'.format(sum(households_no_head_equal == False)))
```

    0 Households with no head have different Target value.
    


```python
#Lets fix the data
#Set poverty level of the members and the head of the house within a family.
# Iterate through each household
for household in not_equal.index:
    # Find the correct label (for the head of household)
    true_target = int(df_income_train[(df_income_train['idhogar'] == household) & (df_income_train['parentesco1'] == 1.0)]['Target'])
    
    # Set the correct label for all members in the household
    df_income_train.loc[df_income_train['idhogar'] == household, 'Target'] = true_target
    
    
# Groupby the household and figure out the number of unique values
all_equal = df_income_train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))
```

    There are 0 households where the family members do not all have the same target.
    

### 3.3 Lets check for any bias in the dataset


```python
#Lets look at the dataset and plot head of household and Target
# 1 = extreme poverty 2 = moderate poverty 3 = vulnerable households 4 = non vulnerable households 
target_counts = heads['Target'].value_counts().sort_index()
target_counts
```




    1     222
    2     442
    3     355
    4    1954
    Name: Target, dtype: int64




```python
target_counts.plot.bar(figsize = (8, 6),linewidth = 2,edgecolor = 'k',title="Target vs Total_Count")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ac071a8cf8>




![png](/images/IncomeQualification_files/IncomeQualification_57_1.png)



```python
# extreme poverty is the smallest count in the train dataset. The dataset is biased.
```

#### 3.4 Lets look at the Squared Variables
'SQBescolari'  
'SQBage'  
'SQBhogar_total'  
'SQBedjefe'  
'SQBhogar_nin'  
'SQBovercrowding'  
'SQBdependency'  
'SQBmeaned'  
'agesq'  


```python
#Lets remove them
print(df_income_train.shape)
cols=['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 
        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']


for df in [df_income_train, df_income_test]:
    df.drop(columns = cols,inplace=True)

print(df_income_train.shape)
```

    (9557, 143)
    (9557, 134)
    


```python
id_ = ['Id', 'idhogar', 'Target']

ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone']

ind_ordered = ['rez_esc', 'escolari', 'age']

hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']

hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
```


```python
#Check for redundant household variables
heads = df_income_train.loc[df_income_train['parentesco1'] == 1, :]
heads = heads[id_ + hh_bool + hh_cont + hh_ordered]
heads.shape
```




    (2973, 98)




```python
# Create correlation matrix
corr_matrix = heads.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

to_drop
```




    ['coopele', 'area2', 'tamhog', 'hhsize', 'hogar_total']




```python
corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>r4t3</th>
      <th>tamhog</th>
      <th>tamviv</th>
      <th>hhsize</th>
      <th>hogar_total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>r4t3</th>
      <td>1.000000</td>
      <td>0.996884</td>
      <td>0.929237</td>
      <td>0.996884</td>
      <td>0.996884</td>
    </tr>
    <tr>
      <th>tamhog</th>
      <td>0.996884</td>
      <td>1.000000</td>
      <td>0.926667</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>tamviv</th>
      <td>0.929237</td>
      <td>0.926667</td>
      <td>1.000000</td>
      <td>0.926667</td>
      <td>0.926667</td>
    </tr>
    <tr>
      <th>hhsize</th>
      <td>0.996884</td>
      <td>1.000000</td>
      <td>0.926667</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>hogar_total</th>
      <td>0.996884</td>
      <td>1.000000</td>
      <td>0.926667</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9],
            annot=True, cmap = plt.cm.Accent_r, fmt='.3f');
```


![png](/images/IncomeQualification_files/IncomeQualification_65_0.png)



```python
# There are several variables here having to do with the size of the house:
# r4t3, Total persons in the household
# tamhog, size of the household
# tamviv, number of persons living in the household
# hhsize, household size
# hogar_total, # of total individuals in the household
# These variables are all highly correlated with one another.
```


```python
cols=['tamhog', 'hogar_total', 'r4t3']
for df in [df_income_train, df_income_test]:
    df.drop(columns = cols,inplace=True)

df_income_train.shape
```




    (9557, 131)




```python
#Check for redundant Individual variables
ind = df_income_train[id_ + ind_bool + ind_ordered]
ind.shape
```




    (9557, 39)




```python
# Create correlation matrix
corr_matrix = ind.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

to_drop
```




    ['female']




```python
# This is simply the opposite of male! We can remove the male flag.
for df in [df_income_train, df_income_test]:
    df.drop(columns = 'male',inplace=True)

df_income_train.shape
```




    (9557, 130)




```python
#lets check area1 and area2 also
# area1, =1 zona urbana 
# area2, =2 zona rural 
#area2 redundant because we have a column indicating if the house is in a urban zone

for df in [df_income_train, df_income_test]:
    df.drop(columns = 'area2',inplace=True)

df_income_train.shape
```




    (9557, 129)




```python
#Finally lets delete 'Id', 'idhogar'
cols=['Id','idhogar']
for df in [df_income_train, df_income_test]:
    df.drop(columns = cols,inplace=True)

df_income_train.shape
```




    (9557, 127)



## Step 4: Predict the accuracy using random forest classifier.


```python
x_features=df_income_train.iloc[:,0:-1]
y_features=df_income_train.iloc[:,-1]
print(x_features.shape)
print(y_features.shape)
```

    (9557, 126)
    (9557,)
    


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report

x_train,x_test,y_train,y_test=train_test_split(x_features,y_features,test_size=0.2,random_state=1)
rmclassifier = RandomForestClassifier()
```


```python
rmclassifier.fit(x_train,y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=10,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)




```python
y_predict = rmclassifier.predict(x_test)
```


```python
print(accuracy_score(y_test,y_predict))
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))
```

    0.9267782426778243
    [[ 124    7    2   24]
     [   1  276    4   36]
     [   0   12  183   38]
     [   2    9    5 1189]]
                  precision    recall  f1-score   support
    
               1       0.98      0.79      0.87       157
               2       0.91      0.87      0.89       317
               3       0.94      0.79      0.86       233
               4       0.92      0.99      0.95      1205
    
        accuracy                           0.93      1912
       macro avg       0.94      0.86      0.89      1912
    weighted avg       0.93      0.93      0.92      1912
    
    


```python
y_predict_testdata = rmclassifier.predict(df_income_test)
```


```python
y_predict_testdata
```




    array([4, 4, 4, ..., 4, 4, 2], dtype=int64)



## Step 6: Check the accuracy using random forest with cross validation.


```python
from sklearn.model_selection import KFold,cross_val_score
```

#### 6.1 Checking the score using default 10 trees


```python
seed=7
kfold=KFold(n_splits=5,random_state=seed,shuffle=True)

rmclassifier=RandomForestClassifier(random_state=10,n_jobs = -1)
print(cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy'))
results=cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy')
print(results.mean()*100)
```

    [0.93514644 0.93043933 0.92569335 0.9314495  0.9314495 ]
    93.08356268159017
    

#### 6.2 Checking the score using 100 trees


```python
num_trees= 100

rmclassifier=RandomForestClassifier(n_estimators=100, random_state=10,n_jobs = -1)
print(cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy'))
results=cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy')
print(results.mean()*100)
```

    [0.94246862 0.94979079 0.94557823 0.94243851 0.94976452]
    94.60081361157272
    


```python
y_predict_testdata = rmclassifier.predict(df_income_test)
y_predict_testdata
```




    array([4, 4, 4, ..., 4, 4, 4], dtype=int64)



#### Looking at the accuracy score, RandomForestClassifier with cross validation has the highest accuracy score of 94.60%

To get a better sense of what is going on inside the RandomForestClassifier model, 
lets visualize how our model uses the different features and which features have greater effect.


```python
rmclassifier.fit(x_features,y_features)
labels = list(x_features)
feature_importances = pd.DataFrame({'feature': labels, 'importance': rmclassifier.feature_importances_})
feature_importances=feature_importances[feature_importances.importance>0.015]
feature_importances.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>v2a1</td>
      <td>0.018653</td>
    </tr>
    <tr>
      <th>2</th>
      <td>rooms</td>
      <td>0.025719</td>
    </tr>
    <tr>
      <th>9</th>
      <td>r4h2</td>
      <td>0.020706</td>
    </tr>
    <tr>
      <th>10</th>
      <td>r4h3</td>
      <td>0.019808</td>
    </tr>
    <tr>
      <th>11</th>
      <td>r4m1</td>
      <td>0.015271</td>
    </tr>
  </tbody>
</table>
</div>




```python
feature_importances.sort_values(by=['importance'], ascending=True, inplace=True)
feature_importances['positive'] = feature_importances['importance'] > 0
feature_importances.set_index('feature',inplace=True)
feature_importances.head()

feature_importances.importance.plot(kind='barh', figsize=(11, 6),color = feature_importances.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')
```




    Text(0.5, 0, 'Importance')




![png](/images/IncomeQualification_files/IncomeQualification_91_1.png)


From the above figure, meaneduc,dependency,overcrowding has significant influence on the model.
