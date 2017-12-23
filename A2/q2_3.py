'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    for i in range(train_data.shape[0]):
        for j in range(64):
            if train_data[i][j] == 1:
                eta[int(train_labels[i])][j] += 1

    eta = (eta +1)/((train_data.shape[0]/10)+2)

    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    image = []
    for i in range(10):
        img_i = class_images[i]
        image.append(img_i.reshape(8,8))

    all_concat = np.concatenate(image, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    for i in range(eta.shape[0]):
        for j in range(eta.shape[1]):
            generated_data[i][j] = np.random.binomial(1,eta[i][j])
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    g_lh = np.zeros((bin_digits.shape[0],10))
    for i in range(bin_digits.shape[0]):
        for k in range(10):
            temp = 1
            for j in range(64):
                first = eta[k][j]**bin_digits[i][j]
                sec = (1-eta[k][j])**(1-bin_digits[i][j])
                temp = temp*first*sec
            g_lh[i][k] = temp
    return np.log(g_lh)

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gl = np.exp(generative_likelihood(bin_digits, eta))
    p_y = 0.1
    p_x = np.sum(gl * p_y, axis=1).reshape(-1, 1)
    return np.log(gl) + np.log(p_y) - np.log(p_x)

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    total = 0
    # Compute as described above and return
    for i in range(bin_digits.shape[0]):
        correct_class = labels[i]
        total += cond_likelihood[i][int(correct_class)]
    return total / bin_digits.shape[0]

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    classes = np.zeros((bin_digits.shape[0], 1))
    for i in range(bin_digits.shape[0]):
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
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)
    #print(eta)
    # Evaluation
    plot_images(eta)
    generate_new_data(eta)
    avg_train_clh = avg_conditional_likelihood(train_data,train_labels, eta)
    avg_test_clh = avg_conditional_likelihood(test_data, test_labels, eta)
    print("train: %s test: %s" %(avg_test_clh, avg_train_clh))
    train_classes = classify_data(train_data,eta)
    train_accuracy = eval_accuracy(train_classes, train_labels)
    test_classes = classify_data(test_data, eta)
    test_accuracy = eval_accuracy(test_classes, test_labels)

    print("train_accruracy: %s test_accuract: %s" %(train_accuracy, test_accuracy))


if __name__ == '__main__':
    main()
