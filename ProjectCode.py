# Ankush Mukhedkar (Email: amuk8580@gmail.com)
# For ShapeAI boot camp project

#mount the drive as the file is in google collab folder
#from google.colab import drive  
#drive.mount('/content/drive') 
#Check if the drive has been mounted
#!ls

#Import required libraries
import pandas
import numpy

#read the contents of CSV file and create a data frame out of it
df = pandas.DataFrame(pandas.read_csv('/content/drive/My Drive/ColabNotebooks/train (1).csv'))
df.head()
df.shape

nCols = df.isnull().sum() #counts the number of null values per column

drop_col = nCols[nCols > ((df.shape[0]) * 35/100)]  #get columns which are having null values above 35% of total row count
#drop_col
#drop_col.index
df.drop(drop_col.index, axis=1, inplace=True)
df.isnull().sum()
df.corr()

df.fillna(df.mean(), inplace=True)
df.isnull().sum()

df['Embarked'].describe()  #see the count, unique, top, frequency of top
df['Embarked'].fillna('S',inplace=True)  #fill-in the blanks with the value for top frequency
df.isnull().sum()   #There shouldn't be any nulls by now
df.corr()   #check the corelation now
 
df['FamilySize'] = df['SibSp']+df['Parch']
df.drop(['SibSp','Parch'], axis=1, inplace=True)
df.corr()
 
df['Alone'] = [0 if df['FamilySize'][i]>0 else 1 for i in df.index]
df.groupby(['Alone'])['Survived'].mean()
df[['Alone','Fare']].corr()
 
df['Sex']= [0 if df['Sex'][i]=='male' else 1 for i in df.index]
df.groupby(['Sex'])['Survived'].mean()
 
df.groupby(['Embarked'])['Survived'].mean()
df.corr()

