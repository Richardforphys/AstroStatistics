from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay as CMD
import plot_settings

#Load data
digits = datasets.load_digits()

train_sizes = np.linspace(1,9,9) * 0.1
print(train_sizes)
ACC         = np.zeros_like(train_sizes)

for i,ts in enumerate(train_sizes):
    
    print('Train size', ts)

    #Separate data for test and prediction
    X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, train_size=ts)

    #Logistic regression fit
    LR = LogisticRegression(solver='sag')
    clf = LR.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    #Compute and print Accuracy value
    ACC[i] = accuracy_score(Y_test, Y_pred)

plt.scatter(train_sizes, ACC, marker='x', color='red')
plt.title('Accuracy vs Train size')
plt.xlabel('Train size percentage')
plt.ylabel('Accuracy')
plt.show()