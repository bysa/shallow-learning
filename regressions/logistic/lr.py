from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


data = pd.read_csv('iris-data.csv')
data.head()
data.describe()
data.info()

# drop na values
data = data.dropna()
data.info()

# make classes consistent
data['class'].value_counts()
data['class'].replace(["Iris-setossa", "versicolor"],
                      ["Iris-setosa", "Iris-versicolor"], inplace=True)
data['class'].value_counts()

# select 2 categories only
final_data = data[data['class'] != 'Iris-virginica']
final_data['class'].value_counts()

# sns.pairplot(final_data, hue='class', height=2.5)
# plt.show()


final_data.hist(column='sepal_length_cm', bins=20, figsize=(10, 5))
# remove outliests
final_data.loc[final_data.sepal_length_cm < 1, [
    'sepal_length_cm']] = final_data['sepal_length_cm']*100
final_data.hist(column='sepal_length_cm', bins=20, figsize=(10, 5))
final_data = final_data.drop(final_data[(
    final_data['class'] == "Iris-setosa") & (final_data['sepal_width_cm'] < 2.5)].index)


# sns.pairplot(final_data, hue='class', size=2.5)
# plt.show()

# replace class names with values
final_data['class'].replace(
    ["Iris-setosa", "Iris-versicolor"], [1, 0], inplace=True)
final_data.head()

# model construction
# get values for matrix
inp_data = final_data.iloc[:, :-1].values
out_data = final_data.iloc[:, -1].values

# scale
scalar = StandardScaler()
inp_data = scalar.fit_transform(inp_data)

# split into test and train
X_train, X_test, y_train, y_test = train_test_split(
    inp_data, out_data, test_size=0.2, random_state=42)

# X_tr_arr = X_train
# X_ts_arr = X_test
# y_tr_arr = y_train.as_matrix()
# y_ts_arr = y_test.as_matrix()
print('Input Shape', (X_train.shape))
print('Output Shape', X_test.shape)


def weight_init(n_features):
    w = np.zeros((1, n_features))
    b = 0
    return w, b


def sigmoid_activation(result):
    return 1/(1+np.exp(-result))


def model_optimize(w, b, X, Y):
    m = X.shape[0]

    # Prediction
    final_result = sigmoid_activation(np.dot(w, X.T)+b)
    Y_T = Y.T
    cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) +
                          ((1-Y_T)*(np.log(1-final_result)))))

    # Gradient calculation
    dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T))
    db = (1/m)*(np.sum(final_result-Y.T))

    grads = {"dw": dw, "db": db}

    return grads, cost


def model_predict(w, b, X, Y, learning_rate, no_iterations):
    costs = []
    for i in range(no_iterations):
        #
        grads, cost = model_optimize(w, b, X, Y)
        #
        dw = grads["dw"]
        db = grads["db"]
        # weight update
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)
        #

        if (i % 100 == 0):
            costs.append(cost)
            print("Cost after %i iteration is %f" % (i, cost))

    # final parameters
    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}

    return coeff, gradient, costs


def predict(final_pred, m):
    y_pred = np.zeros((1, m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
    return y_pred


# Get number of features
n_features = X_train.shape[1]
print('Number of Features', n_features)
w, b = weight_init(n_features)
# Gradient Descent
coeff, gradient, costs = model_predict(
    w, b, X_train, y_train, learning_rate=0.0001, no_iterations=4500)
# Final prediction
w = coeff["w"]
b = coeff["b"]
print('Optimized weights', w)
print('Optimized intercept', b)
#
final_train_pred = sigmoid_activation(np.dot(w, X_train.T)+b)
final_test_pred = sigmoid_activation(np.dot(w, X_test.T)+b)
#
m_tr = X_train.shape[0]
m_ts = X_test.shape[0]
#
y_tr_pred = predict(final_train_pred, m_tr)
print('Training Accuracy', accuracy_score(y_tr_pred.T, y_train))
#
y_ts_pred = predict(final_test_pred, m_ts)
print('Test Accuracy', accuracy_score(y_ts_pred.T, y_test))


plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title('Cost reduction over time')
plt.show()


# Sklearn
clf = LogisticRegression()
clf.fit(X_tr_arr, y_tr_arr)
print(clf.intercept_, clf.coef_)
pred = clf.predict(X_ts_arr)
print('Accuracy from sk-learn: {0}'.format(clf.score(X_ts_arr, y_ts_arr)))
