#%% Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
#%% Dataframe
df = pd.read_csv('casas.csv')
#%%
columns = ['tamanho', 'ano', 'garagem']
#%% Creating X and y
X = df.drop('preco', axis=1)
y = df['preco']
#%% Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

#%% Model
model = LinearRegression()
model.fit(X_train, y_train)
model.predict([[120,2001,2]])
#%% Model Dumping
pickle.dump(model, open('house_quotes.sav', 'wb'))
#%%
