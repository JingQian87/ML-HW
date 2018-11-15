import math
import os
import random as rnd
import re
from abc import abstractmethod, ABC
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from nltk.stem.snowball import SnowballStemmer


class Metrics(object):
    # L1 metric
    @staticmethod
    def l1(x1, x2):
        d = np.array([x1, x2])
        distance = 0
        for v in d:
            distance += abs(v[0] - v[1])
        return distance

    # L2 metric
    @staticmethod
    def l2(x1, x2):
        d = np.array([x1, x2])
        distance = 0
        for v in d:
            distance += pow((v[0] - v[1]), 2)
        return distance

    # L  metric
    @staticmethod
    def l_inf(x1, x2):
        d = np.array([x1, x2])
        distance = - float('inf')
        difference = 0
        for v in d:
            difference = abs(v[0] - v[1])
            if difference > distance:
                distance = difference
        return difference


class Classifier(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def train(self, train_X, train_Y):
        '''

        :param train_X: m * n np 2d array. m is the dataset size, n is the feature size
        :param train_Y: m * 1 np 2d array. m is the dataset size
        :return: prediction accuracy
        '''
        pass

    @abstractmethod
    def predict(self, x):
        '''

        :param x: a single data input
        :return: the predicted label the the single data input x
        '''
        pass

    @abstractmethod
    def evaluate(self, test_X, test_Y):
        '''

        :param test_X: m * n np 2d array. m is the dataset size, n is the feature size
        :param test_Y: m * 1 np 2d array. m is the dataset size
        :return: prediction accuracy
        '''

        pass


class EmailsDataset(object):
    def __init__(self):
        self.classes = {}
        self.files_directory = ""
        self.words = {}

    # read data without labels
    def read_emails(self):
        data = []
        for key, value in self.classes.items():
            for file in value:
                with open(os.path.join(self.files_directory, key, file), 'r', errors="replace", encoding='utf-8') as f:
                    f_read = f.read()
                    data.append(f_read)
        return data

    # read data from certain class
    def read_emails_class(self, c):
        files = []
        class_emails = self.classes[c]
        for file in class_emails:
            with open(os.path.join(self.files_directory, c, file), 'r', errors="replace", encoding='utf-8') as f:
                f_read = f.read()
                files.append(f_read)
        return files

    # read labeled data
    def read_emails_y(self):
        data = {}
        for key in self.classes:
            data[key] = self.read_emails_class(key)
        return data

    # Split data into training and testing
    def load_and_split_data(self, ratio):
        training_data = {}
        testing_data = {}
        data = self.read_emails_y()
        length = 0
        for c in data:
            length += len(data[c])

        min_length = length
        for c in data:
            len_c = len(data[c])
            if len_c < min_length:
                min_length = len_c
        # TODO: make testing size equal
        for c in data:
            c_train_size = int(min_length * ratio)
            c_training = []
            c_testing = data[c].copy()
            while len(c_training) < c_train_size:
                i = rnd.randrange(len(c_testing))
                c_training.append(c_testing.pop(i))
            training_data[c] = c_training
            testing_data[c] = c_testing

        training_x, training_y, testing_x, testing_y = self.prepare_bags_of_words(training_data, testing_data)
        return [training_x, training_y, testing_x, testing_y]

    # Split data into training and testing
    def prepare_bags_of_words(self, training_data, testing_data):
        text = []
        for c in training_data:
            text.extend(training_data[c])
        dictionary = self.create_dictionary(text)
        self.words = dictionary
        training_x = []
        training_y = []
        for c in training_data:
            for file in training_data[c]:
                bag = self.create_bag_of_words(sentence=file, dictionary=dictionary)
                training_x.append(bag)
                if c == "spam":
                    training_y.append(1)
                else:
                    training_y.append(0)

        testing_x = []
        testing_y = []
        for c in testing_data:
            for file in testing_data[c]:
                bag = self.create_bag_of_words(sentence=file, dictionary=dictionary)
                testing_x.append(bag)
                if c == "spam":
                    testing_y.append(1)
                else:
                    testing_y.append(0)

        shuffled = list(zip(training_x, training_y))
        rnd.shuffle(shuffled)
        training_x, training_y = zip(*shuffled)

        shuffled = list(zip(testing_x, testing_y))
        rnd.shuffle(shuffled)
        testing_x, testing_y = zip(*shuffled)

        return [np.array(training_x), np.array(training_y), np.array(testing_x), np.array(testing_y)]

    # Embed text data in  Euclidean space:
    # 1. Read Text.
    def read_files(self, files_directory):
        classes = {}
        for f in sorted(os.listdir(files_directory)):
            folder_path = os.path.join(files_directory, f)
            if os.path.isdir(folder_path):
                file_names = []
                for d in sorted(os.listdir(folder_path)):
                    if os.path.join(folder_path, d):
                        file_names.append(d)
                classes[f] = file_names
        self.classes = classes
        self.files_directory = files_directory
        return self

    # 2. Clean and Stem words.
    @staticmethod
    def stem_words(sentence):
        words = re.compile('[a-zA-Z]+').findall(sentence)
        stem = SnowballStemmer("english")
        stem_words = []
        for word in words:
            stem_words.append(stem.stem(word))
        return stem_words

    # 3. Count unique words and order them alphabetically.
    def create_dictionary(self, training_data):
        words = {}
        if type(training_data) is list:
            text = "".join(training_data)
        elif type(training_data) is str:
            text = training_data
        else:
            return "error not supported type "

        file = self.stem_words(text)
        i = 0
        for stemmed in file:
            if stemmed not in words:
                words[stemmed] = i
                i += 1
        return words

    # 4. transform text data into vectors "bag of words"
    def create_bag_of_words(self, sentence, dictionary=None):
        if dictionary is None:
            dictionary = self.words
        bag_of_words = np.zeros(len(dictionary))
        stem_words = self.stem_words(sentence)
        for stemmed in stem_words:
            if stemmed in dictionary:
                bag_of_words[dictionary[stemmed]] += 1
        return bag_of_words


class NaiveBayes(Classifier):
    def __init__(self):
        super().__init__()
        self.prior = {}
        self.conditional_p = {}

    def train(self, train_x, train_y):
        n = train_x.shape[1]  # vocabulary size
        m = train_x.shape[0]  # training size
        n1 = np.count_nonzero(train_y)
        n0 = m - np.count_nonzero(train_y)
        self.prior[1] = n1 / m
        self.prior[0] = n0 / m
        training_words_0 = np.zeros(n)
        training_words_1 = np.zeros(n)
        i = 0
        while i < m:
            if train_y[i] == 0:
                training_words_0 = np.sum([train_x[i], training_words_0], axis=0)
            else:
                training_words_1 = np.sum([train_x[i], training_words_1], axis=0)
            i += 1

        j = 0
        count_0 = sum(training_words_0)
        count_1 = sum(training_words_1)
        while j < n:
            self.conditional_p[(j, 0)] = (training_words_0[j] + 1) / (count_0 + n)
            self.conditional_p[(j, 1)] = (training_words_1[j] + 1) / (count_1 + n)
            j += 1

        return self

    def predict(self, x):
        score = {}
        max_c = ""
        max_score = float('-inf')
        for c in self.prior:
            score[c] = math.log(self.prior[c])
            i = 0
            while i < len(x):
                score[c] += x[i] * math.log(self.conditional_p[(i, c)])
                i += 1
            if score[c] > max_score:
                max_score = score[c]
                max_c = c
        return max_c

    def predictions(self, test_x):
        predictions = []
        for x in test_x:
            predictions.append(self.predict(x))
        return predictions

    def evaluate(self, test_x, test_y):
        predictions = self.predictions(test_x)
        total = 0
        correct = 0
        for p, y in zip(predictions, test_y):
            if p == y:
                correct += 1
            total += 1
        return correct / total


class NearestNeighbor(Classifier):
    def __init__(self, K=1, metric=Metrics.l1):
        super().__init__()
        self.K = K
        self.training_data = np.array([])
        self.metric = metric

    def train(self, train_X, train_Y):
        self.training_data = list(zip(train_X, train_Y))
        return self

    def predict(self, x):
        distances = []
        for xn in self.training_data:
            distances.append((self.metric(xn[0], x), xn))

        distances = sorted(distances, key=itemgetter(0))
        y = {0: 0, 1: 0}
        k = 0
        while k < self.K:
            row = distances[k]
            point = row[1]
            y[point[1]] += 1
            k += 1
        return max(y, key=y.get)

    def predictions(self, test_x):
        predictions = []
        for x in test_x:
            predictions.append(self.predict(x))
        return predictions

    def evaluate(self, test_x, test_y, metric=Metrics.l1):
        predictions = self.predictions(test_x)
        total = 0
        correct = 0
        for p, y in zip(predictions, test_y):
            if p == y:
                correct += 1
            total += 1

        accuracy = correct / total
        return accuracy


def data_preprocessing(reload=False, ratio=0.9):
    if reload:

        # Prepare data:
        emails = EmailsDataset()
        emails.read_files(files_directory="./enron1")
        training_x, training_y, testing_x, testing_y = emails.load_and_split_data(ratio)

        # save the split data so we don't need to load every time.
        np.save('./enron1/training_x', training_x)
        np.save('./enron1/training_y', training_y)
        np.save('./enron1/testing_x', testing_x)
        np.save('./enron1/testing_y', testing_y)
    else:
        training_x = np.load("./enron1/training_x.npy")
        training_y = np.load("./enron1/training_y.npy")
        testing_x = np.load("./enron1/testing_x.npy")
        testing_y = np.load("./enron1/testing_y.npy")

    return training_x, training_y, testing_x, testing_y


# Main function:
if __name__ == "__main__":
    from DecisionTree import DecisionTree

    # Prepare data:
    ratios = [0.50, 0.60, 0.80, 0.90]
    accuracies_nn = []
    accuracies_nb = []
    accuracies_dt = []
    for ratio in ratios:
        training_x, training_y, testing_x, testing_y = data_preprocessing(reload=True, ratio=ratio)
        print("Files ready!")
        # Call NB model and do prediction:
        model_nb = NaiveBayes()
        model_nb.train(training_x, training_y)
        accuracy_nb = model_nb.evaluate(testing_x, testing_y)
        accuracies_nb.append(accuracy_nb)

        # Call NN model and do prediction:
        model_nn = NearestNeighbor(K=3, metric=Metrics.l2)
        model_nn.train(training_x, training_y)
        accuracy_nn = model_nn.evaluate(testing_x, testing_y)
        accuracies_nn.append(accuracy_nn)

        # Call Decision Tree model and do prediction:
        model_dt = DecisionTree(max_tree_len=10, train_Y=training_y)
        model_dt.train(training_x, training_y)
        accuracy_dt = model_dt.evaluate(testing_x, testing_y)
        accuracies_dt.append(accuracy_dt)

    plt.plot(ratios, accuracies_nb, label="Accuracy of Naive Bayes Classifier")
    plt.plot(ratios, accuracies_nn, label="Accuracy of K-Nearest Neighbor Classifier")
    plt.plot(ratios, accuracies_dt, label="Accuracy of Decision Tree Classifier")
    plt.xlabel('training size')
    plt.ylabel('accuracy')
    plt.title('Accuracy comparison at different training size')
    plt.legend()
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig("accuracy_graph.png")
