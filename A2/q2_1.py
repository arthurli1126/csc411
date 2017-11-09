'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
from sklearn.model_selection import KFold

# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        distances = self.l2_distance(test_point)
        # if k == 1:
        #     index = np.where(distances==np.min(distances))
        #     print(type(index))
        #     digit = self.train_labels[index]
        #     print("k: %s major: %s" % (k, digit))
        # else:
        digit,j = 0,0
        total_index = distances.argsort()
        while True:
            index = total_index[:k-j]
            labels = []
            for i in range(index.size):
                labels.append(self.train_labels[index[i]])
            labels = [int(i) for i in labels]
            temp_digit,count = np.unique(np.array(labels),return_counts=True)
            counts = dict(zip(temp_digit,count))
            major_count = max(counts.values())
            major_digit = [k for k,v in counts.items() if v == major_count]
            if len(major_digit)==1:
                digit = major_digit[0]
                print("k: %s major: %s" %(k-j,major_digit))
                break
            j = j+1

        return digit

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    kf = KFold(n_splits=10)
    average_acc = []
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        accuracies = []
        for train_index, test_index in kf.split(train_data):
            knn = KNearestNeighbor(train_data[train_index], train_labels[train_index])
            accuracies.append(classification_accuracy(knn,
                                                 k,
                                                 train_data[test_index],
                                                 train_labels[test_index]))
        average_acc.append(sum(accuracies)/float(len(accuracies)))
    return average_acc




def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''

    match = 0
    index = 0
    for i in eval_data:
        knn_label = knn.query_knn(i,k)
        print("knn labbel %s" %knn_label)
        if knn_label == eval_labels[index]:
            match = match + 1
        index = index+1
    return match/len(eval_data)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)
    # print(test_data[0])
    # print(test_labels[0])
    # Example usage:
    #predicted_label = knn.query_knn(test_data[0], 3)
    #print(predicted_label)
    print(classification_accuracy(knn,1,test_data,test_labels))
    print(classification_accuracy(knn,15, test_data,test_labels))
    print(cross_validation(train_data,train_labels))


if __name__ == '__main__':
    main()