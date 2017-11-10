'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        means[i] = means[i] + np.mean(i_digits, axis=0)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)
    # Compute covariances
    for k in range(10):
        i_digits = data.get_digits_by_label(train_data, train_labels, k)
        for j in range(64):
            for i in range(64):
                covariances[k][j][i] = np.mean((i_digits[:,j] - means[k][j])*(i_digits[:,i] - means[k][i]))
                if (i==j):
                    covariances[k][j][i] = covariances[k][j][i] + 0.01
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    cov_diag = np.zeros((10,8,8))
    for i in range(10):
        cov_diag[i] = np.diag(covariances[i]).reshape(8,8)

    all_concat = np.concatenate(np.log(cov_diag), 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    g_likelihood = np.zeros((digits.shape[0],10))
    for k in range(10):
        for i in range(digits.shape[0]):
            first_part = (2*np.pi)**(-32)*(np.linalg.det(covariances[k])**(-0.5))
            sec = -0.5* (digits[i]- means[k]).transpose()
            third = np.linalg.inv(covariances[k])
            final = first_part * np.exp(np.dot(sec,third).dot(digits[i] - means[k]))
            g_likelihood[i][k] = np.log(final)
    return g_likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    return None

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    return None

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # print(means)
    # print(covariances)
    #plot_cov_diagonal(covariances)

    g_lh = generative_likelihood(train_data,means,covariances)

    print(g_lh)
    # Evaluation

if __name__ == '__main__':
    main()