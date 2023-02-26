''' Phishing Domain Detection using Machine Learning with Python.
 This is the basic working of the Program.
 Required Libraries: sklearn and pandas.
'''
# Import Reading Library
import pandas as pd

# Read the Data. This is the Dataset we will be working with.
data = pd.read_csv("Training Dataset.csv")

''' Split Data into X and Y. Look at X and Y as X to be input and Y to be output.
 So we're basically splitting our data into input and output for processing reasons.'''
X = data.iloc[:,:-1] # This refers to all columns but the last one. Input.
Y = data.iloc[:,-1]  # This refers to the last column. Output.

''' Import the Train/Test Library. This Library helps us split our data into Training and Testing Data.
 Our Algorithm will Train on the 70% of the data (Train Data) and perform predictions on 30% the (Test Data).
Hence, test_size on Line 23 = 0.3'''
from sklearn.model_selection import train_test_split

# Split data into training and testing Data.
x_train, x_test, y_train, y_test = \
    train_test_split(X,Y, test_size = 0.3, random_state=1234)

''' We are going to perform the classification using Decision Trees.    
Perform Decision Tree Classification. This is where the magic(predictions) happen.'''
from sklearn import tree

dtree = tree.DecisionTreeClassifier() # Initialize The Classifier.
dtree.fit(x_train, y_train)           # Perform Training(Fitting) on the training Data.
prediction = dtree.predict(x_test)    # Make predictions on the Input Test Data.

''' Calculate Accuracy. This tells us how well our model Performed Overall. 
In our case it is able to detect 95-96/100 Phishing Domains which isn't too bad.'''
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(prediction, y_test) * 100 # This based our Predictions and Known correct values
print("Accuracy is ", accuracy)
