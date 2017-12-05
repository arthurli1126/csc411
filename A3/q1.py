'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier


class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''

    def __init__(self, data, targets, batch_size=200):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = self.data[indices]
        y_batch = self.targets[indices]
        return X_batch, y_batch



def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)
    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model



def mnb_model(train_tf_idf, train_labels, test_tf_idf, test_labels):
    # training the model
    sampler = BatchSampler(train_tf_idf, train_labels, 4000)
    model = MultinomialNB()
    space = np.linspace(0.1,10,50)
    param_grid = dict(alpha=space)
    grid = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    sample_X, sample_y = sampler.get_batch()
    grid.fit(sample_X, sample_y)
    print("best alpha is {}, and accuracy is {}".format(grid.best_params_, grid.best_score_))
    model = grid.best_estimator_
    print("param {}".format(model.get_params()))
    # evaluate the model
    train_prediction = model.predict(train_tf_idf)
    test_prediction = model.predict(test_tf_idf)

    print('multinomial train accuracy = %s' %((train_prediction== train_labels).mean()))
    print('multinomial test accuracy = %s' % ((test_prediction == test_labels).mean()))

    return model



#this is a trash model for this task
def knn_model(train_data, train_labels, test_data, test_labels):

    knn_model = KNeighborsClassifier(kernel='linear')
    k_range = range(1, 100)
    param_grid = dict(n_neighbors=k_range)
    grid = RandomizedSearchCV(knn_model, param_grid, n_iter=10, cv=10, scoring='accuracy', n_jobs=5)
    grid.fit(train_data, train_labels)
    print("The best hypeparameter of the knn model is %s, ".format('KNN', grid.best_params_, grid.best_score_))
    knn_model = grid.best_estimator_
    train_pred = knn_model.predict(train_data)
    print('KNN train accuracy  = {}'.format((train_pred == train_labels).mean()))
    test_pred = knn_model.predict(test_data)
    accuracy=(test_pred == test_labels).mean()
    print('KNN test accuracy= {}'.format(accuracy))

    return knn_model,accuracy


def svm_model(train_data, train_labels, test_data, test_labels):

    sampler = BatchSampler(train_tf_idf, train_labels, 500)
    SVM  = svm.SVC()
    c_range = np.linspace(0.001, 10, 50)
    param_grid = dict(C=c_range)
    grid = GridSearchCV(svm_model, param_grid, cv=10, scoring='accuracy', n_jobs=4)
    sample_X, sample_y = sampler.get_batch()
    grid.fit(sample_X, sample_y)
    print("best alpha is {}, and accuracy is {}".format(grid.best_params_, grid.best_score_))
    SVM=grid.best_estimator_
    # SVM.fit(train_data,train_labels)
    # train_pred = SVM.predict(train_data)
    print('SVM train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = SVM.predict(test_data)
    accuracy=(test_pred == test_labels).mean()
    print('SVM test accuracy = {}'.format( accuracy))

    return SVM,accuracy

def random_forest(train_data, train_labels, test_data, test_labels):
    # sampler = BatchSampler(train_tf_idf, train_labels, 500)
    RFC = RandomForestClassifier(n_estimators=30)
    # c_range = np.linspace(0.001, 10, 50)
    # param_grid = dict(C=c_range)
    # grid = GridSearchCV(svm_model, param_grid, cv=10, scoring='accuracy', n_jobs=4)
    # sample_X, sample_y = sampler.get_batch()
    # grid.fit(sample_X, sample_y)
    # print("best alpha is {}, and accuracy is {}".format(grid.best_params_, grid.best_score_))
    # SVM=grid.best_estimator_
    RFC.fit(train_data, train_labels)
    train_pred = RFC.predict(train_data)
    print('SVM train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = RFC.predict(test_data)
    accuracy = (test_pred == test_labels).mean()
    print('SVM test accuracy = {}'.format(accuracy))

    return RFC


def confusion_matrix(pre_labels, test_labels):
    k = test_labels.shape[0]
    c_matrix = np.zeros((k,k))
    for i in range(k):
        if(pre_labels[i]==test_labels[i]):
                c_matrix[i][i] = c_matrix[i][i] + 1





if __name__ == '__main__':
    train_data, test_data = load_data()

    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    train_tf_idf,test_tf_idf, feature_names = tf_idf_features(train_data, test_data)
    print(train_bow.shape)
    print(test_bow.shape)
    print(train_tf_idf.shape)
    #print(train_tf_idf[0])
    print(test_tf_idf.shape)
    #bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    #mnb_model = mnb_model_1(train_tf_idf, train_data.target, test_tf_idf, test_data.target)
    #lol = mnb_model(train_tf_idf, train_data.target, test_tf_idf, test_data.target)
    #knn = knn_model(train_tf_idf, train_data.target, test_tf_idf, test_data.target)
    #svm_model,svm_acc = svm_model(train_tf_idf, train_data.target, test_tf_idf, test_data.target)
    RFC = random_forest(train_tf_idf, train_data.target, test_tf_idf, test_data.target)



