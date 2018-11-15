from sklearn.datasets import load_iris
from sklearn import tree
from email_spam_classification import data_preprocessing

dt = tree.DecisionTreeClassifier(criterion='entropy',  max_depth=10)
training_x, training_y, testing_x, testing_y = data_preprocessing(False)
dt.fit(training_x, training_y)
correct = 0

for i in range(len(testing_x)):
    predict_y = dt.predict(testing_x[i].reshape(1, -1))
    correct += testing_y[i] == predict_y

accuracy = correct / len(training_x)
print(accuracy)
tree.export_graphviz(dt, 'graph')