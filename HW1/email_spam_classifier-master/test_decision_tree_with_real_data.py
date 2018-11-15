from email_spam_classification import data_preprocessing
from DecisionTree import DecisionTree
import pickle
import time

training_x, training_y, testing_x, testing_y = data_preprocessing(False)
print('data loaded.')
start_time = time.time()
dt = DecisionTree(max_tree_len=10, train_Y=training_y)
dt.train(training_x, training_y)

with open('./model/dt', 'wb') as file:
    pickle.dump(dt, file)


with open('./model/dt', 'rb') as file:
    dt2 = pickle.load(file)

print(dt2.evaluate(training_x, training_y))
print(dt2.evaluate(testing_x, testing_y))


