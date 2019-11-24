from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model

boston_house_data = datasets.load_iris()
# print (boston_house_data)

x_input, x_test, y_input, y_test = model_selection.train_test_split(boston_house_data['data'], boston_house_data['target'])

model = linear_model.LogisticRegression()
model.fit(x_input, y_input)
score = model.score(x_test, y_test)
predict = model.predict(x_test)

print(score)