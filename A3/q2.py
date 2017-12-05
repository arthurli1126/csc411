import numpy as np

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)


class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''

    def __init__(self, data, targets, batch_size):
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
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch


class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        params[0] = (-1)*self.lr*grad + params[0]*self.beta
        params[1] = params[1] + params[0]
        return params


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count, b = 0):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        self.b = b

    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        highe_vector = []
        temp = 1-y*(np.dot(X,self.w))
        for i in range(temp.shape[0]):
            if temp[i]>0:
                highe_vector.append(temp[i])
            else:
                highe_vector.append(0)
        return highe_vector

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        norm = self.w
        sec = self.c/y.shape[0]
        loss = self.hinge_loss(X, y)
        sub_grad = np.zeros(X.shape[1])
        for i in range(len(loss)):
            if(loss[i]!=0):
                sub_grad = sub_grad - y[i]*X[i]

        return norm+(sec*sub_grad)


    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        return np.where(np.dot(X,self.w)>0,1,-1)


def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets



def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.
    '''
    batch = BatchSampler(train_data, train_targets, batchsize)
    SVM_train = SVM(penalty,train_data.shape[1])
    param = [np.zeros_like(SVM_train.w),SVM_train.w]
    for i in range(iters):
        X,y =  batch.get_batch()
        grad = SVM_train.grad(X,y)
        param = optimizer.update_params(param,grad)
        SVM_train.w = param[1]
    return SVM_train

def sgdm_verif():
    sgdm = GDOptimizer(1)
    grad = 0.01 * 2 * 10.0
    param = [0.0, 10.0]
    params = []
    for i in range(200):
        param = sgdm.update_params(param, grad)
        grad = 0.01 * 2 * param[1]
        params.append(param[1])

    sgdm_2 = GDOptimizer(1, 0.9)
    grad_2 = 0.01 * 2 * 10.0
    param_2 = [0.0, 10.0]
    params_2 = []
    for j in range(200):
        param_2 = sgdm_2.update_params(param_2, grad_2)
        grad_2 = 0.01 * 2 * param_2[1]
        params_2.append(param_2[1])

    plt.plot(params, 'r', params_2, 'b')
    plt.show()






if __name__ == '__main__':
    #sgdm_verif()
    train_data, train_targets, test_data, test_targets = load_data()
    train_data = np.concatenate((np.ones((train_data.shape[0],1)),train_data),axis=1)
    test_data = np.concatenate((np.ones((test_data.shape[0],1)),test_data),axis=1)
    optimizer = GDOptimizer(0.05)
    svm = optimize_svm(train_data,train_targets,1.0,optimizer,100,500)
    train_result_zero = svm.classify(train_data)
    train_acuracy_zero = np.where(train_result_zero == train_targets, 1, 0).mean()
    train_loss_zero = svm.hinge_loss(train_data,train_targets)

    test_result_zero = svm.classify(test_data)
    test_acuracy_zero = np.where(test_result_zero == test_targets, 1, 0).mean()
    test_loss_zero = svm.hinge_loss(test_data, test_targets)

    print("train accuracy with B = 0 %s" %train_acuracy_zero)
    #print("train loss with B = 0 %s" %train_loss_zero)

    print("train accuracy with B = 0 %s" % test_acuracy_zero)
    #print("train loss with B = 0 %s" % test_loss_zero)

    #print(svm.w.shape)


    plt.imshow(svm.w[1:].reshape(28,28), cmap= 'gray')
    plt.show()



    optimizer = GDOptimizer(0.05,0.1)
    svm = optimize_svm(train_data, train_targets, 1.0, optimizer, 100, 500)
    train_result_1 = svm.classify(train_data)
    train_acuracy_1 = np.where(train_result_1 == train_targets, 1, 0).mean()
    train_loss_1 = svm.hinge_loss(train_data, train_targets)

    test_result_1 = svm.classify(test_data)
    test_acuracy_1 = np.where(test_result_1 == test_targets, 1, 0).mean()
    test_loss_1 = svm.hinge_loss(test_data, test_targets)

    print("train accuracy with B = 0.1 %s" % train_acuracy_1)
    # print("train loss with B = 0.1 %s" %train_loss_1)

    print("train accuracy with B = 0.1 %s" % test_acuracy_1)
    # print("train loss with B = 0.1 %s" % test_loss_zero)

    plt.imshow(svm.w[1:].reshape(28, 28), cmap='gray')
    plt.show()












