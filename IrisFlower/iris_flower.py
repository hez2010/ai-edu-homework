from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model

iris_flower = datasets.load_iris()
# print (iris_flower)

x_input, x_test, y_input, y_test = model_selection.train_test_split(iris_flower['data'], iris_flower['target'])

model = linear_model.LogisticRegression()
model.fit(x_input, y_input)
score = model.score(x_test, y_test)
predict = model.predict(x_test)

print(score)