
# Song Prediction Model

This machine learning model predicts whether the cateory of song is Rock or Hip-Hop. 

The **main.py** implements various Algorithms on the data. These Algorithms include:

- **LogisticRegression**
- **GuassianNB**
- **Decision_Tree**
- **KNeighborsClassifier**
- **Neural Network**

After comparing the result of all these algorithms, **KNeighborsClassifier** seems to be the most suitable Algorithm. The **Decision_Tree_Classifier** overfitted since it gave 0.0 error on training data although the error on Cross_Validation and Training data was also low.

These comparisons are written in the file **result.txt**. The **prediction.py** then imports the KNeighborsClassifier algorithm and make prediction on the data provided by the user.





## Deployment

To use the project, run the following command in the folder where you want to get the project.

```bash
  git clone git@github.com:anamfatima1304/Songs_Prediction.git
```
Now play around with the model and enjoy.


## Installation

To run this project, you should have python istalled on your computer. Also install the following packages

```bash
  pip install pandas
  pip install numpy
  pip install seaborn
  pip install sklearn
  pip install tensorflow
```
    
## Code Guide

Import the following packages

```bash
  import pandas as pd
  import numpy as np
  import seaborn as sns
  from sklearn.preprocessing import LabelEncoder
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.naive_bayes import GaussianNB
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.metrics import mean_squared_error
  from sklearn.tree import DecisionTreeClassifier
  import tensorflow as tf
```

Now let's move to read the dataset
```bash
  audio_data = pd.read_csv("audio_data.csv")
  audio_data.drop(['Unnamed: 0'], inplace=True, axis = 1)
```
After reading the dataset, we took a look at how does the dataset look and find out if there is any null  alue:

```bash
  audio_data.head()
  audio_data.describe()
  audio_data.isnull.sum()
```

It comes out that there is no NULL value but the genre is in the form of Rock and Hip-Hop. So we need to encode the data as 0 and 1.

```bash
  encoder = LabelEncoder()
  audio_data["genre_top"] = encoder.fit_transform(audio_data["genre_top"])
```

It labels Hip_Hop as zero and Rock as One. After that we found co_relation for **Feature Reduction** but there was no such features to be dropped out.

```bash
   #  finding the co-relation for Feature Selection
  audio_data_features = audio_data[audio_data.columns[:-1]]
  corelation = audio_data_features.corr()

  # Plotting the co-relation on a graph
  sns.heatmap(corelation, annot= True)

  # we will drop the columns that have more then 70% corelation
  corr = set()
  threshold = 0.7
  for i in range(len(corelation)):
      for j in range(len(corelation)):
          if abs(corelation.iloc[i,j])>threshold and corelation.iloc[i,j]!=1:
              corr.add(corelation.columns[i])
  # print(corr)
  #  As the set is empty, we get to know that there is no string relationship in the features.
```

After that, it is the time to divide the data in dependant and independant features

```bash
  X = audio_data_features
  y = audio_data["genre_top"]
```

Now using train_test_split to split the data into training, cross_validation and testing.

```bash
  X_train, X_ , y_train , y_ = train_test_split(X, y , train_size=0.6, random_state=42)
  X_cv, X_test, y_cv, y_test = train_test_split(X_, y_ , train_size=0.5, random_state=42)
```

Lets scale the data using StandardScaler

```bash
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_cv = scaler.transform(X_cv)
  X_test = scaler.transform(X_test)
```


Now, after the data is in its processed form, its time to apply the Algorithms. the Next part applies all the Algorithms and save their error and number of mis-classified Examples in the text file.

The prediction.py then deals with predicting the values provided by the user.
## Contributing

Contributions are always welcome!

Find out an Error or give any suggestion to improve the project.

If you want to add any new features, make a pull request.

ThankYou for your attention.

