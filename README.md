<H3>NAME : Shaik Sameer Basha</H3>
<H3>REGISTER NO. 212222240093</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 22/08/2024 </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

### To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
### Hardware – PCs
### Anaconda – Python 3.7 Installation / Google Colab /Jupyter Notebook

## RELATED THEORETICAL CONCEPT:

### Kaggle:

Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

### Data preprocessing:


Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

### Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
### STEP 1:

#### Importing the libraries<BR>

### STEP 2:

#### Importing the dataset<BR>

### STEP 3:

#### Taking care of missing data<BR>

### STEP 4:

#### Encoding categorical data<BR>

### STEP 5:

#### Normalizing the data<BR>


### STEP 6:

Splitting the data into test and train<BR>

##  PROGRAM:

### Import Libraries
```py

from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```

### Read the dataset 

```py
df=pd.read_csv("Churn_Modelling.csv")
```
### Checking Data
```py
df.head()
df.tail()
df.columns
```

### Check the missing data
```py
df.isnull().sum()
```

### Check for Duplicates
```py
df.duplicated()
```

### Assigning Y
```py
y = df.iloc[:, -1].values
print(y)
```

### Check for duplicates
```py
df.duplicated()
```

### Check for outliers
```py
df.describe()
```

### Dropping string values data from dataset
```py
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```
### Checking datasets after dropping string values data from dataset
```py
data.head()
```

### Normalize the dataset
```py
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```

### Split the dataset
```py
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
```

### Training and testing model
```py
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```
## OUTPUT:

### Data checking
![1 1](https://github.com/user-attachments/assets/1034cd57-80c2-4261-8269-585ab7640497)



### Missing Data 

![1 2](https://github.com/user-attachments/assets/c5bad26b-df7e-4171-a083-7f69782d4b35)


### Duplicates identification

![1 3](https://github.com/user-attachments/assets/1efe63f0-371f-4fcc-a2f6-129492e416e1)


### Vakues of 'Y'


![1 4](https://github.com/user-attachments/assets/03badcf6-4eb4-4710-b861-9b568909215e)

### Outliers

![1 5](https://github.com/user-attachments/assets/3d9b9421-0619-4b8d-94f4-ad513c3bd742)


### Checking datasets after dropping string values data from dataset
![1 6](https://github.com/user-attachments/assets/93a38c94-a8dd-4855-af4f-c30b515959f9)



### Normalize the dataset
![1 7](https://github.com/user-attachments/assets/d1a13238-4e00-4599-a7dd-8fd21b169914)


### Split the dataset

![1 8](https://github.com/user-attachments/assets/81ad9134-1086-490f-a4db-4fbb1cf24471)


### Training and testing model

![1 9](https://github.com/user-attachments/assets/f6b0b3ac-3150-4d20-b3b8-1094f33a4d1a)


## RESULT:

### Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
