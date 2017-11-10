from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.scatter([j[i] for j in X],y,s=5)
        plt.title('%s vs PRICE' %features[i])

    plt.tight_layout(pad=1.08,h_pad=None,w_pad=None,rect=None)
    plt.show()


def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    return np.linalg.solve(np.dot(X.transpose(),X), np.dot(X.transpose(),Y))

def predict_housing_price(w,X):
    return np.dot(X,w)

def mse(y_fit, y):
    mse_value = 0
    for i in range(y_fit.size):
        mse_value += (y[i] - y_fit[i])**2
    return mse_value/y_fit.size

def mae(y_fit, y):
    mae_value = 0
    for i in range(y_fit.size):
        mae_value += np.sqrt((y[i] - y_fit[i])**2)
    return mae_value/y_fit.size

def msle(y_fit, y):
    mae_value = 0
    for i in range(y_fit.size):
        mae_value += np.sqrt((y[i] - y_fit[i])**2)
    return np.log(mae_value/y_fit.size)

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))

    # Visualize the features
    visualize(X, y, features)


    #TODO: Split data into train and test
    list_x = X.tolist()
    new_list_X = [[1]+i  for i in list_x]
    X = np.array(new_list_X)
    testset_index = np.random.choice(505,int(505*0.2),replace=False)
    test_set_x = np.array([X[i] for i in testset_index])
    train_set_x = np.array([X[i] for i in range(505) if i not in testset_index])
    test_set_y = np.array([y[i] for i in testset_index])
    train_set_y = np.array([y[i] for i in range(505) if i not in testset_index])
    #print "total: %s" %(test_set+train_set).szie
    # Fit regression model
    #add bias term

    w = fit_regression(train_set_x, train_set_y)

    print(w)
    # Compute fitted values, MSE, etc.
    y_fit = predict_housing_price(w,test_set_x)
    print(y_fit)
    print(test_set_y)
    print(mse(y_fit,test_set_y))
    print(mae(y_fit, test_set_y))
    print(msle(y_fit, test_set_y))



if __name__ == "__main__":
    main()

