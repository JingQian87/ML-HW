from email_spam_classification import Classifier
import numpy as np


class DecisionTree(Classifier):

    def __init__(self, max_tree_len, train_Y):
        """

        :param max_tree_len: the tree will stop growing once its length reaches max_tree_len
        :param train_Y: we need train_Y to get the all possible labels.
        """
        self.max_tree_len = max_tree_len
        self.root = DecisionTreeNode(0, max_tree_len, set(train_Y))

    def train(self, train_X, train_Y):
        m, n = train_X.shape
        self.root.train(train_X=train_X, train_Y=train_Y, attrs_ids=list(range(n)))

    def predict(self, x):
        return self.root.predict(x)

    def evaluate(self, test_x, test_y):
        correct = 0
        for i in range(len(test_x)):

            predict_y = self.root.predict(test_x[i])
            correct += test_y[i] == predict_y

        accuracy = correct/len(test_x)
        return accuracy


class DecisionTreeNode(object):

    def __init__(self, tree_level, max_tree_len, Y_value_set):
        self.tree_level = tree_level
        self.max_tree_len = max_tree_len
        self.children = []

        self.node_label = None
        self.threhold = None
        self.split_attrs_idx = None

        self.is_leaf = False

        self.Y_value_set = Y_value_set

    def train(self, train_X, train_Y, attrs_ids):

        # stop condition
        if np.all(train_Y == train_Y[0], axis=0):
            self.node_label = train_Y[0]
            self.is_leaf = True
            return

        if len(train_X) == 0:
            raise Exception("This node runs out of data")

        if len(attrs_ids) == 0:
            self.node_label = self.majority(train_Y)
            self.is_leaf = True
            return

        if self.tree_level == self.max_tree_len:
            self.node_label = self.majority(train_Y)
            self.is_leaf = True
            return

        parent_entropy = self.compute_entropy(train_Y)
        left_train_X, left_train_Y, left_attrs_ids, right_train_X, right_train_Y, right_attrs_ids, self.split_attrs_idx, self.threhold = \
            self.split_data_and_attrs(train_X, train_Y, attrs_ids, parent_entropy)

        left_tree = DecisionTreeNode(tree_level=self.tree_level + 1, max_tree_len=self.max_tree_len,
                                     Y_value_set=self.Y_value_set)
        right_tree = DecisionTreeNode(tree_level=self.tree_level + 1, max_tree_len=self.max_tree_len,
                                      Y_value_set=self.Y_value_set)

        if len(left_train_X) == 0:
            left_tree.node_label = self.majority(train_Y)
        else:
            left_tree.train(train_X=left_train_X, train_Y=left_train_Y, attrs_ids=left_attrs_ids)

        if len(right_train_X) == 0:
            right_tree.node_label = self.majority(train_Y)
        else:
            right_tree.train(train_X=right_train_X, train_Y=right_train_Y, attrs_ids=right_attrs_ids)

        self.children.append(left_tree)
        self.children.append(right_tree)


    def majority(self, train_Y):
        """
        Find The most frequent layer in the given dataset.
        :rtype: object
        :param train_Y:
        :return: The most frequent layer in the given dataset.
        """
        vals, counts = np.unique(train_Y, return_counts=True)
        i = np.argmax(counts)
        return vals[i]

    def split_data_and_attrs(self, train_X, train_Y, attrs_ids, parent_entropy):
        # return
        # for each attribute, find the best value to split.
        max_info_gain = 0
        best_threshold = train_X[0, attrs_ids[0]]  # first point in the dataset

        best_attr_idx_i = 0  # index of elements in best_attr_idx
        best_attr_idx = attrs_ids[best_attr_idx_i]  # best attribute's idx

        for i, idx in enumerate(attrs_ids):
            column = train_X[:, idx]  # column is a vector of all data entries of one attribute.

            _set = set(column)
            for entry in _set:
                info_gain = self.compute_info_gain(column=column, threshold=entry,
                                                   train_Y=train_Y, parent_entropy=parent_entropy)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_threshold = entry
                    best_attr_idx_i = i
                    best_attr_idx = idx

        left_train_X = train_X[train_X[:, best_attr_idx] < best_threshold]
        left_train_Y = train_Y[train_X[:, best_attr_idx] < best_threshold]  # Use train_X's indices to select Y

        right_train_X = train_X[train_X[:, best_attr_idx] >= best_threshold]
        right_train_Y = train_Y[train_X[:, best_attr_idx] >= best_threshold]  # Use train_X's indices to select Y

        left_attrs_ids = attrs_ids[:best_attr_idx_i] + attrs_ids[best_attr_idx_i + 1:]
        right_attrs_ids = attrs_ids[:best_attr_idx_i] + attrs_ids[best_attr_idx_i + 1:]

        return left_train_X, left_train_Y, left_attrs_ids, right_train_X, right_train_Y, right_attrs_ids, \
               best_attr_idx, best_threshold

    def split_data_and_attrs_optimized_by_compute_percentile(self, train_X, train_Y, attrs_ids, parent_entropy):
        # return
        # for each attribute, find the best value to split.
        max_info_gain = 0
        best_threshold = train_X[0, attrs_ids[0]]  # first point in the dataset
        best_attr_idx_i = 0  # index of elements in best_attr_idx
        best_attr_idx = attrs_ids[best_attr_idx_i]  # best attribute's idx

        for i, idx in enumerate(attrs_ids):
            column = train_X[:, idx]  # column is a vector of all data entries of one attribute.

            for percentile in range(0, 100, 15):
                entry = np.percentile(column, percentile)
                info_gain = self.compute_info_gain(column=column, threshold=entry,
                                                   train_Y=train_Y, parent_entropy=parent_entropy)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_threshold = entry
                    best_attr_idx_i = i
                    best_attr_idx = idx

        left_train_X = train_X[train_X[:, best_attr_idx] <= best_threshold]
        left_train_Y = train_Y[train_X[:, best_attr_idx] <= best_threshold]  # Use train_X's indices to select Y

        right_train_X = train_X[train_X[:, best_attr_idx] >= best_threshold]
        right_train_Y = train_Y[train_X[:, best_attr_idx] >= best_threshold]  # Use train_X's indices to select Y

        left_attrs_ids = attrs_ids[:best_attr_idx_i]
        right_attrs_ids = attrs_ids[best_attr_idx_i + 1:]

        return left_train_X, left_train_Y, left_attrs_ids, right_train_X, right_train_Y, right_attrs_ids, \
               best_attr_idx, best_threshold

    def split_data_and_attrs_optimized_by_attrs_entropy(self, train_X, train_Y, attrs_ids, parent_entropy):

        # for each attribute, find the best value to split.
        max_info_gain = 0
        best_threshold = train_X[0, attrs_ids[0]]  # first point in the dataset

        attrs_entropies = np.apply_along_axis(self.compute_entropy_for_attributes, arr=train_X.take(attrs_ids, axis = 1), axis=0)
        i = np.argmax(attrs_entropies)
        best_attr_idx_i = i
        best_attr_idx = attrs_ids[i]

        column = train_X[:, best_attr_idx]  # column is a vector of all data entries of one attribute.

        for column_i, entry in enumerate(column):
            info_gain = self.compute_info_gain(column=column, threshold=entry,
                                               train_Y=train_Y, parent_entropy=parent_entropy)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_threshold = entry

        left_train_X = train_X[train_X[:, best_attr_idx] < best_threshold]
        left_train_Y = train_Y[train_X[:, best_attr_idx] < best_threshold]  # Use train_X's indices to select Y

        right_train_X = train_X[train_X[:, best_attr_idx] >= best_threshold]
        right_train_Y = train_Y[train_X[:, best_attr_idx] >= best_threshold]  # Use train_X's indices to select Y

        left_attrs_ids = attrs_ids[:best_attr_idx_i]
        right_attrs_ids = attrs_ids[best_attr_idx_i + 1:]

        return left_train_X, left_train_Y, left_attrs_ids, right_train_X, right_train_Y, right_attrs_ids, best_attr_idx, best_threshold

    def compute_info_gain(self, column, threshold, train_Y, parent_entropy):

        left_Y = train_Y[column < threshold]
        right_Y = train_Y[column >= threshold]
        left_entropy = (len(left_Y) / len(train_Y)) * self.compute_entropy(left_Y) if len(left_Y) > 0 else 0
        right_entropy = (len(right_Y) / len(train_Y)) * self.compute_entropy(right_Y) if len(right_Y) > 0 else 0

        info_gain = parent_entropy - left_entropy - right_entropy
        return info_gain

    def compute_entropy(self, x):

        # _set = set(x)
        probs = np.zeros(len(self.Y_value_set))
        for i, y_val in enumerate(self.Y_value_set):
            probs[i] = float(len(x[x == y_val])) / len(x)

        probs = probs[probs != 0]
        return -np.sum(probs * np.log2(probs))

    def compute_entropy_for_attributes(self, x):

        _set = set(x)
        probs = np.zeros(len(_set))
        for i, y_val in enumerate(_set):
            probs[i] = float(len(x[x == y_val])) / len(x)

        probs = probs[probs != 0]
        return -np.sum(probs * np.log2(probs))

    def predict(self, X):
        if self.is_leaf:
            return self.node_label

        if X[self.split_attrs_idx] < self.threhold:
            return self.children[0].predict(X)
        else:
            return self.children[1].predict(X)
