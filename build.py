import pandas as pd
import pickle

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
df = pd.read_csv('C:\\Users\\Hp\\deploy_df')
#df.drop('Unnamed: 0', axis=1, inplace=True)

x = df.drop('Price', axis=1)
y = df['Price']

# splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

print(x_train.head())


ET_Model = ExtraTreesRegressor(n_estimators=120)
ET_Model.fit(x_train, y_train)


predict = ET_Model.predict(x_test)


# # saving model to disk
pickle.dump(ET_Model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(predict)
