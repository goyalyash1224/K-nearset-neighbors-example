#import basic libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


// Read the 'KNN_Project_Data csv file into a dataframe 

df = pd.read_csv('KNN_Project_Data')
df.head() 

//use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.

sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')


// now  standardize the variables.


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))  //fit scaler to features

scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1)) //drop 'TARGET CLASS' from scaled features


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1]) //Convert the scaled features to a dataframe
df_feat.head()

# Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)


from sklearn.neighbors import KNeighborsClassifier  //import kNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)  //fit the knn model for k=1

##Evaluation of model


pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))

# Choosing a K Value
Let's go ahead and use the elbow method to pick a good K Value!

error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

//making a plot as error_rate vs k
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

## Retrain with new K Value
knn = KNeighborsClassifier(n_neighbors=30) //here  k=30 for knn model

knn.fit(X_train,y_train)
pred = knn.predict(X_test)


print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

# Thanku
