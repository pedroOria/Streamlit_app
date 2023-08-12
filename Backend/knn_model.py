import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

class IrisModel():
  def __init__(self, dataset):
    self.dataset = dataset
    self.model = None
  def train(self, k_value):
    X = self.dataset.drop(['Id','Species'], axis=1)
    y = self.dataset['Species']
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7, random_state=5)
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_train,y_train)
    y_predict = knn.predict(X_test)
    accuracy = metrics.accuracy_score(y_predict, y_test)
    self.model = knn
    return accuracy
  def predict(self,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm):
    pred_result = self.model.predict([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])[0]
    return pred_result
