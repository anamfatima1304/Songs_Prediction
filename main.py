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

def mis_classified(original, predicted):
    # Counting the number of values that has been mis-classified 
    value = 0
    for i in range(len(original)):
        if(original[i]!=predicted[i]):
            value+=1
    return value


def print_metrics(data, error, missed = 0):
    # Printing the error and the number of the values that has been misclassified
    print("\t", data)
    print("\t\t Mean Squared Error:", error)
    print("\t\t Missed value:", missed)


def process_data():
    # Reading the file
    audio_data = pd.read_csv("audio_data.csv")
    audio_data.drop(['Unnamed: 0'], inplace=True, axis = 1)

    # Checking the file
    # audio_data.head()

    # audio_data.describe()

    #  Checking if there is any null value
    # audio_data.isnull.sum()

    # Encoding the "genre_top" as 0 and 1
    encoder = LabelEncoder()
    audio_data["genre_top"] = encoder.fit_transform(audio_data["genre_top"])
    # It labels Hip_Hop as zero and Rock as One

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

    X = audio_data_features
    y = audio_data["genre_top"]

    # Now, Our data is ready for the train test splitting
    X_train, X_ , y_train , y_ = train_test_split(X, y , train_size=0.6, random_state=42)
    X_cv, X_test, y_cv, y_test = train_test_split(X_, y_ , train_size=0.5, random_state=42)

    # Scaling the Data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_cv = scaler.transform(X_cv)
    X_test = scaler.transform(X_test)

    y_train = np.array(y_train)
    y_cv = np.array(y_cv)
    y_test = np.array(y_test)

    return X_test, y_test, X_cv, y_cv, X_train, y_train

def Prediction_Model():

    X_test, X_cv, y_test, y_cv, X_train, y_train = process_data()

    # Using Different Algorithms to predict the data
    logistic_Regression = LogisticRegression()
    GuassianNB = GaussianNB()
    kneighbours = KNeighborsClassifier(n_neighbors=10)
    Decision_Tree = DecisionTreeClassifier()
    ml_Algorithms = [logistic_Regression, GuassianNB, Decision_Tree, kneighbours]

    file = open ("result.txt", "w")
    for i in ml_Algorithms:
        model = i
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        error1 = mean_squared_error(y_train, y_pred)
        missed1 = mis_classified(y_train, y_pred)
        
        y_pred = model.predict(X_cv)
        error2 = mean_squared_error(y_cv, y_pred)
        missed2 = mis_classified(y_cv, y_pred)
        
        y_pred = model.predict(X_test)
        error3 = mean_squared_error(y_test, y_pred)
        missed3 = mis_classified(y_test, y_pred)
        
        print(i,":")
        print_metrics("Training data", error1, missed1)
        print_metrics("Cross_Validation data", error2, missed2)
        print_metrics("Testing data", error3, missed3)
        file.write(f"{i}\n{error1}    {error2}    {error3}    {missed1}    {missed2}    {missed3} \n")
        print("\n\n")

    data = X_train
    # Using Neural Networks to predict the song
    model = tf.keras.Sequential([
        tf.keras.Input((None, 2881, 9)),
        tf.keras.layers.Dense(units=1000, activation="relu"),
        tf.keras.layers.Dense(units=500, activation="relu"),
        tf.keras.layers.Dense(units=100, activation="relu"),
        tf.keras.layers.Dense(units=1, activation="softmax")]
    )

    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy())

    history = model.fit(X_train, y_train, epochs=100)

    y_pred = model.predict(X_train)
    error1 = mean_squared_error(y_train, y_pred)
    missed1 = mis_classified(y_train, y_pred)

    y_pred = model.predict(X_cv)
    error2 = mean_squared_error(y_cv, y_pred)
    missed2 = mis_classified(y_cv, y_pred)

    y_pred = model.predict(X_test)
    error3 = mean_squared_error(y_test, y_pred)
    missed3 = mis_classified(y_test, y_pred)

    print("Neural Network",":")
    print_metrics("Training data", error1, missed1)
    print_metrics("Cross_Validation data", error2, missed2)
    print_metrics("Testing data", error3, missed3)
    file.write(f"Neural Network\n{error1}    {error2}    {error3}    {missed1}    {missed2}    {missed3} \n")

    file.close()

def final_algorithm():
    # After analyzing all the models, KNeighborsClassifier seems to be the most suitable one
    X_test, y_test, X_cv, y_cv, X_train, y_train = process_data()
    kneighbours = KNeighborsClassifier(n_neighbors=10)
    model = kneighbours
    model.fit(X_train, y_train)
    return model

if __name__=="__main__":
    Prediction_Model()