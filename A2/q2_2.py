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

# def cov_compute_sigma_mles(train_data, train_labels):
#     '''
#     Compute the covariance estimate for each digit class
#
#     Should return a three dimensional numpy array of shape (10, 64, 64)
#     consisting of a covariance matrix for each digit class
#     '''
#     covariances = np.zeros((10, 64, 64))
#     # Compute covariances
#     for k in range(10):
#         i_digits = data.get_digits_by_label(train_data, train_labels, k)
#         covariances[k] = np.cov(i_digits.T)
#     return covariances



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
            # if final>1:
            #     print("final: %s first: %s sec: %s third %s" %(final,first_part,sec,third))
            g_likelihood[i][k] = np.log(final)
    return g_likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gl = np.exp(generative_likelihood(digits,means, covariances))
    p_y = 0.1
    p_x = np.sum(gl*p_y, axis=1).reshape(-1,1)
    return np.log(gl) + np.log(p_y) - np.log(p_x)

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    total = 0
    # Compute as described above and return
    for i in range(digits.shape[0]):
        correct_class = labels[i]
        total += cond_likelihood[i][int(correct_class)]

    return total/digits.shape[0]

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    classes = np.zeros((digits.shape[0],1))
    for i in range(digits.shape[0]):
        classes[i] = np.where(cond_likelihood[i] == cond_likelihood[i].max())
    return classes

def eval_accuracy(predicted_labels, true_labels):
    match = 0
    for i in range(predicted_labels.shape[0]):
        if int(predicted_labels[i])== int(true_labels[i]):
            match+=1
    return match/predicted_labels.shape[0]

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    #cov = cov_compute_sigma_mles(train_data, train_labels)
    # print(means)
    # print(np.max(covariances))
    # print(np.max(cov))
    # print(np.linalg.det(cov)**-0.5)
    plot_cov_diagonal(covariances)

    #g_lh = generative_likelihood(train_data,means,covariances)
    #c_lh = conditional_likelihood(train_data,means,covariances)
    #print(np.sum(g_lh))
    #print(np.sum(np.exp(c_lh)))

    avg_test_clh = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print("average conditional likelihood test: %s" %(avg_test_clh))

    avg_train_clh = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    print("average conditional likelihood train: %s" % (avg_train_clh))

    # Evaluation

    classes = classify_data(train_data, means, covariances)

    print("train accuracy: %s" %eval_accuracy(classes, train_labels))

    test_classes = classify_data(test_data, means, covariances)

    print("test accuracy: %s"  %eval_accuracy(test_classes, test_labels))

if __name__ == '__main__':
    main()