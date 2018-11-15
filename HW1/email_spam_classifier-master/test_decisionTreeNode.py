from unittest import TestCase
from DecisionTree import DecisionTreeNode
import numpy as np


class TestDecisionTreeNode(TestCase):
    def test_train(self):
        print('start test test_split_data_and_attrs')
        # def split_data_and_attrs(self, train_X, train_Y, attrs_ids, parent_entropy):
        train_X = np.array([[1, 2, 3, 1],
                            [1, 3, 5, 8],
                            [1, 2, 7, 5]]
                           )
        train_Y = np.array([0, 0, 1])
        attrs_ids = range(4)

        node = DecisionTreeNode(1, 3, set(train_Y))
        left_train_X, left_train_Y, left_attrs_ids, right_train_X, right_train_Y, right_attrs_ids = \
            node.split_data_and_attrs(train_X=train_X, train_Y=train_Y, attrs_ids=attrs_ids, parent_entropy=1)
        print('case 1:')
        node.train(train_X=train_X, train_Y=train_Y, attrs_ids=attrs_ids)

        print('--------------')

    def test_majority(self): # pass
        return
        print('--------')
        print('case 1:')
        train_Y = np.array([5, 0, 0, 5, 0, 1]) # majority 0
        node = DecisionTreeNode(1, 3, set(train_Y))
        print(node.majority(train_Y))
        print('--------')

        print('--------')
        print('case 2:')
        train_Y = np.array([5, 0, 5, 5, 0, 1]) # majority 5
        node = DecisionTreeNode(1, 3, set(train_Y))
        print(node.majority(train_Y))
        print('--------')

        pass

    def test_split_data_and_attrs_optimized(self): # pass
        return
        train_X = np.array([[1, 2,3],
                            [1, 3,5],
                            [1, 2,7]]
                           )
        train_Y = np.array([0, 0, 1])
        node = DecisionTreeNode(1, 3, set(train_Y))
        left_train_X, left_train_Y, left_attrs_ids, right_train_X, right_train_Y, right_attrs_ids = \
            node.split_data_and_attrs_optimized(train_X=train_X, train_Y=train_Y, attrs_ids=[0, 1, 2], parent_entropy=1)
        print('case 1:')
        print('left_train_X:')
        print(left_train_X)
        print(left_train_Y)
        print('\nright_train_X')
        print(right_train_X)
        print(right_train_Y)

        print('--------------')

        print('case 2:')
        train_X = np.array([[1, 1, 1],
                            [1, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0]
                            ]
                           )
        train_Y = np.array([1, 1, 2, 2])
        node = DecisionTreeNode(1, 3, set(train_Y))
        left_train_X, left_train_Y, left_attrs_ids, right_train_X, right_train_Y, right_attrs_ids = \
            node.split_data_and_attrs_optimized(train_X=train_X, train_Y=train_Y, attrs_ids=[0, 1, 2], parent_entropy=1)
        print('\nleft_train_X:')
        print(left_train_X)
        print(left_train_Y)

        print('\nright_train_X')
        print(right_train_X)
        print(right_train_Y)

        print('\nleft_attrs_ids:')
        print(left_attrs_ids)

        print('\nright_attrs_ids')
        print(right_attrs_ids)

        print('--------------')

        print('case 3:')
        train_X = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]
                            ]
                           )
        train_Y = np.array([1, 1, 1, 1])
        node = DecisionTreeNode(1, 3, set(train_Y))
        attrs_ids = [0, 2]
        left_train_X, left_train_Y, left_attrs_ids, right_train_X, right_train_Y, right_attrs_ids = \
            node.split_data_and_attrs_optimized(train_X=train_X, train_Y=train_Y, attrs_ids=attrs_ids, parent_entropy=1)
        print('\nleft_train_X:')
        print(left_train_X)
        print(left_train_Y)

        print('\nright_train_X')
        print(right_train_X)
        print(right_train_Y)

        print('\nleft_attrs_ids:')
        print(left_attrs_ids)

        print('\nright_attrs_ids')
        print(right_attrs_ids)

        print('--------------')
        print('case 5:')
        train_X = np.array([[1, 2,3],
                            [1, 3,5],
                            [1, 2,7]]
                           )
        train_Y = np.array([1, 0, 1])
        node = DecisionTreeNode(1, 3, set(train_Y))
        left_train_X, left_train_Y, left_attrs_ids, right_train_X, right_train_Y, right_attrs_ids = \
            node.split_data_and_attrs_optimized(train_X=train_X, train_Y=train_Y, attrs_ids=[0, 1, 2], parent_entropy=1)

        print('left_train_X:')
        print(left_train_X)
        print(left_train_Y)
        print('\nright_train_X')
        print(right_train_X)
        print(right_train_Y)

        print('--------------')

    def test_split_data_and_attrs(self): # pass
        return
        print('start test test_split_data_and_attrs')
        # def split_data_and_attrs(self, train_X, train_Y, attrs_ids, parent_entropy):
        train_X = np.array([[1, 2,3],
                            [1, 3,5],
                            [1, 2,7]]
                           )
        train_Y = np.array([0, 0, 1])
        node = DecisionTreeNode(1, 3, set(train_Y))
        left_train_X, left_train_Y, left_attrs_ids, right_train_X, right_train_Y, right_attrs_ids = \
            node.split_data_and_attrs(train_X=train_X, train_Y=train_Y, attrs_ids=[0, 1, 2], parent_entropy=1)
        print('case 1:')
        print('left_train_X:')
        print(left_train_X)
        print(left_train_Y)
        print('\nright_train_X')
        print(right_train_X)
        print(right_train_Y)

        print('--------------')

        print('case 2:')
        train_X = np.array([[1, 1, 1],
                            [1, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0]
                            ]
                           )
        train_Y = np.array([1, 1, 2, 2])
        node = DecisionTreeNode(1, 3, set(train_Y))
        left_train_X, left_train_Y, left_attrs_ids, right_train_X, right_train_Y, right_attrs_ids = \
            node.split_data_and_attrs(train_X=train_X, train_Y=train_Y, attrs_ids=[0, 1, 2], parent_entropy=1)
        print('\nleft_train_X:')
        print(left_train_X)
        print(left_train_Y)

        print('\nright_train_X')
        print(right_train_X)
        print(right_train_Y)

        print('\nleft_attrs_ids:')
        print(left_attrs_ids)

        print('\nright_attrs_ids')
        print(right_attrs_ids)

        print('--------------')

        print('case 3:')
        train_X = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]
                            ]
                           )
        train_Y = np.array([1, 1, 1, 1])
        node = DecisionTreeNode(1, 3, set(train_Y))
        attrs_ids = [0, 2]
        left_train_X, left_train_Y, left_attrs_ids, right_train_X, right_train_Y, right_attrs_ids = \
            node.split_data_and_attrs(train_X=train_X, train_Y=train_Y, attrs_ids=attrs_ids, parent_entropy=1)
        print('\nleft_train_X:')
        print(left_train_X)
        print(left_train_Y)

        print('\nright_train_X')
        print(right_train_X)
        print(right_train_Y)

        print('\nleft_attrs_ids:')
        print(left_attrs_ids)

        print('\nright_attrs_ids')
        print(right_attrs_ids)

        print('--------------')
        print('case 4:')
        train_X = np.array([[1, 0],
                            [0, 1]
                            ]
                           )

        train_Y = np.array([1, 1])
        node = DecisionTreeNode(1, 3, set(train_Y))
        attrs_ids = [0, 1]
        left_train_X, left_train_Y, left_attrs_ids, right_train_X, right_train_Y, right_attrs_ids = \
            node.split_data_and_attrs(train_X=train_X, train_Y=train_Y, attrs_ids=attrs_ids, parent_entropy=1)
        print('\nleft_train_X:')
        print(left_train_X)
        print(left_train_Y)

        print('\nright_train_X')
        print(right_train_X)
        print(right_train_Y)

        print('\nleft_attrs_ids:')
        print(left_attrs_ids)

        print('\nright_attrs_ids')
        print(right_attrs_ids)

        print('--------------')
        print('case 5:')
        train_X = np.array([[1, 2, 3],
                            [1, 3, 5],
                            [1, 2, 7]]
                           )
        train_Y = np.array([1, 0, 1])
        node = DecisionTreeNode(1, 3, set(train_Y))
        left_train_X, left_train_Y, left_attrs_ids, right_train_X, right_train_Y, right_attrs_ids = \
            node.split_data_and_attrs(train_X=train_X, train_Y=train_Y, attrs_ids=[0, 1, 2], parent_entropy=1)

        print('left_train_X:')
        print(left_train_X)
        print(left_train_Y)
        print('\nright_train_X')
        print(right_train_X)
        print(right_train_Y)

    print('--------------')
    def test_compute_info_gain(self): # pass
        return
        # compute_info_gain(self, column, column_i, threshold, train_Y, parent_entropy):
        np.random.seed(10)
        column = np.array([10, 5, 2])
        Y = np.array([0, 1, 1])
        node = DecisionTreeNode(1, 3, set(Y))

        print(node.compute_info_gain(column, threshold=3, train_Y=Y, parent_entropy=1))


    def test_compute_entropy(self): # pass
        return
        np.random.seed(10)
        Y = np.random.randint(0, 2, 100)
        node = DecisionTreeNode(1, 3, set(Y))
        print(node.compute_entropy(Y))
