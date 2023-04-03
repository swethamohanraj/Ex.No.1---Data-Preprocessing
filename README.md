# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
```
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train
```
## PROGRAM:
```
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
df = pd.read_csv('Churn_Modelling.csv')
df.head()
le=LabelEncoder()
df["CustomerId"]=le.fit_transform(df["CustomerId"])
df["Surname"]=le.fit_transform(df["Surname"])
df["CreditScore"]=le.fit_transform(df["CreditScore"])
df["Geography"]=le.fit_transform(df["Geography"])
df["Gender"]=le.fit_transform(df["Gender"])
df["Balance"]=le.fit_transform(df["Balance"])
df["EstimatedSalary"]=le.fit_transform(df["EstimatedSalary"])
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
print(df.isnull().sum())
df.fillna(df.mean().round(1),inplace=True)
print(df.isnull().sum())
y=df.iloc[:,-1].values
print(y)
df.duplicated()
print(df['Exited'].describe())
scaler= MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)
x_train,x_test,y_train,x_test=train_test_split(X,Y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:
### Printing first five rows and cols of given dataset:
![image](https://user-images.githubusercontent.com/94228215/229541074-69e3db1d-6076-478f-ae3b-b0599af08633.png)
### Seperating x and y values:
![image](https://user-images.githubusercontent.com/94228215/229542411-6256aaaf-76b9-4da4-be93-382f89c4f034.png)


### Checking NULL value in the given dataset:
![image](https://user-images.githubusercontent.com/94228215/229541986-6e613e82-9b28-4bef-8d6a-3c091514b5bd.png)

### Printing the Y column along with its discribtion:
![image](https://user-images.githubusercontent.com/94228215/229541916-f145d65c-0d08-42ab-9eb9-1ed7512bad8a.png)

### Applyign data preprocessing technique and printing the dataset:
![image](https://user-images.githubusercontent.com/94228215/229541780-7b4eb179-5afa-4910-b0c6-8eadc941ac54.png)


### Printing training set:
![image](https://user-images.githubusercontent.com/94228215/229541705-6a3b51b1-840e-42a6-ab20-4e93c805b2ff.png)

### Printing testing set and length of it:
![image](https://user-images.githubusercontent.com/94228215/229541645-332a54ad-fe9e-4eaa-9a0d-1a6ec79d3eea.png)


## RESULT:
Hence the data preprocessing is done using the above code and data has been splitted into trainning and testing data for getting a better model.

