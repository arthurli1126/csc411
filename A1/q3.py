import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

BATCHES = 50


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


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)


def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    grad = np.zeros(13)

    for i in range(X.shape[0]):
        grad += 2 * (np.dot(w.transpose(), X[i]) - y[i]) * X[i]
    return grad/X.shape[0]

def avg_sgd(w, K, b_s):
    grad = np.zeros(13)
    for i in range(K):
        X_b, y_b = b_s.get_batch()
        grad = lin_reg_gradient(X_b, y_b, w)
        # print(grad)
    return grad/K


def compare(vec1, vec2):
    cos = cosine_similarity(vec1,vec2)
    norm = np.linalg.norm(vec1-vec2)
    return cos,norm


def var_sgd(w, K, b_s):
    var = np.zeros(len(w)*K).reshape(K,len(w))
    for i in range(K):
        X_b, y_b = b_s.get_batch()
        var[i] = lin_reg_gradient(X_b, y_b, w)
    return np.var(var,axis=0)


def var_compare(X,y,w,K):
    va = np.zeros(400*13).reshape(400,13)
    for i in range(400):
        batch_sampler = BatchSampler(X, y, i+1)
        print(i)
        va[i] = var_sgd(w,K,batch_sampler)
        print(i)
    return va





def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)
    # Example usage


    # bath_grad = lin_reg_gradient(X_b, y_b, w)
    # bath_grad = np.zeros(13)
    #
    # for i in range(500):
    #     bath_grad += lin_reg_gradient(X_b, y_b, w)
    #
    # bath_grad = bath_grad/500
    avg_grad = avg_sgd(w,500,batch_sampler)
    true_grad = lin_reg_gradient(X, y, w)
    # print(true_grad)
    # print(avg_grad)
    cos, norm = compare(true_grad,avg_grad)
    print(cos)
    print(norm)
    var = var_compare(X,y,w,500)
    plt.ylabel("log(var)")
    plt.xlabel("log(m)")
    plt.plot(np.log(np.arange(1,401)), np.log(var))
    plt.show()



if __name__ == '__main__':
    main()