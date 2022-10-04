import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle 

data = pd.read_csv("student/student-mat.csv", sep=';')
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
'''
best_acc = 0

for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)


    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)
    print(acc)  # R-Score of our linear regression

    if acc > best_acc:
        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)
'''

pickle_in = open("studentmodel.pickle", 'rb')
linear = pickle.load(pickle_in)

acc = linear.score(x_test, y_test)
print(acc)  # R-Score of our linear regression

print("Co: ", linear.coef_)  # Slope of our coefficients
print("Intercept: ", linear.intercept_)  # Intercept of our model

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'G1'
style.use('ggplot')
plt.scatter(data[p],data['G3'])
plt.xlabel(p)
plt.ylabel('Final Grade')
plt.show()    