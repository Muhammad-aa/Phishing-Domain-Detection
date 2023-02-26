'''
NOTE: The global keywords are only placed there because I like to analyze my variables individually in the Variable Explorer.
The Variable Explorer is available for IDEs like Spyder, Pycharm etc. I use Spyder.
So you can totally remove them (the lines with the global keywords) if you do not need that. The program will still run and print your accuracy to you.
'''
import pandas as pd

'''
This is the Function that does the Magic (Preciction/Classification)
To use the Function: phishing_domain_detector(Name of Dataset file in csv format) as seen on line 46.
'''
def phishing_domain_detector(file=""):   
    # Load the Data
    data = pd.read_csv(file)
   
    # Split Data into X and Y
    global X, Y
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    
    # Split data into Training and Testing Data.
    from sklearn.model_selection import train_test_split
    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = \
        train_test_split(X,Y, test_size=0.3, random_state=1234)
        
    # Perform Decision Tree classsification
    from sklearn import tree
    global dtree
    dtree = tree.DecisionTreeClassifier()
    
   # Handling Exceptions
    try:
        dtree.fit(x_train, y_train)
    except Exception: # This is for When the Data isn't properly Labeled.
       print("[-] Please Ensure your data is properly Labeled\n[-] Exiting...")       
       import sys
       sys.exit()     # Exit the Program if Data isn't properly Labeled.
    else:
        prediction = dtree.predict(x_test)
        
    # Measure our accuracy
    from sklearn.metrics import accuracy_score
    global accuracy
    accuracy = accuracy_score(prediction, y_test) * 100
    print("Accuracy is", accuracy)
    
    
phishing_domain_detector("Training Dataset.csv")