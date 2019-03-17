import pandas as pd
from sklearn import linear_model
import pickle as pkl

# load dataset into a pandas data frame
df = pd.read_csv('winequality-red.csv', delimiter=';')

# take the quality column
labels = df['quality']

# take all the columns except quality
features = df.drop('quality', axis = 1)

regressor = linear_model.LinearRegression()
regressor.fit(features, labels)

print(regressor.predict([[7.4,0.66,0,1.8,0.075,13,40,0.9978,3.51,0.56,9.4]]).tolist())

model_file = open('model.pkl', 'wb')
pkl.dump(regressor, model_file)

model_file = open('model.pkl', 'rb')
loaded_model = pkl.load(model_file)
print(loaded_model.predict([[7.4,0.66,0,1.8,0.075,13,40,0.9978,3.51,0.56,9.4]]).tolist())