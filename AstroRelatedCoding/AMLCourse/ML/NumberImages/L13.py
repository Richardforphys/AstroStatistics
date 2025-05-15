from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay as CMD

#Load data
digits = datasets.load_digits()

#Separate data for test and prediction
X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, train_size=0.9)

#Logistic regression fit
LR = LogisticRegression(solver='sag')
clf = LR.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

#Compute and print Accuracy value
ACC = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {ACC:.2f}')

#Compute and plot confusion matrix
CM = confusion_matrix(Y_test, Y_pred)
disp = CMD(CM, display_labels=clf.classes_)
disp.plot()
plt.grid()
plt.show()

#Plot random images and target
fig, axes = plt.subplots(7,7, figsize=(10, 10))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

#Plot random images, true and predicted labels
fig, axes = plt.subplots(7, 7, figsize=(10, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
mychoices = np.random.choice(len(X_test), 49, replace=False)
for i, ax in enumerate(axes.flat):
    image = X_test[mychoices[i]].reshape(8,8)
    true_label = Y_test[mychoices[i]]
    pred_label = Y_pred[mychoices[i]]
    ax.imshow(image, cmap='binary')
    ax.text(0.05, 0.05, true_label, transform=ax.transAxes, color='green', fontsize=14)
    ax.text(0.85, 0.05, pred_label, transform=ax.transAxes, color='red', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.suptitle('Predicted vs True labels')
plt.show()