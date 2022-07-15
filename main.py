# %% Libraires
# Use this code to get requirements: pip list --format=freeze > requirements.txt

import os
import pickle
from debugpy import listen
from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth # Authentication with username and password
from textblob import TextBlob
from googletrans import Translator #using version 3.1.0a0
from waitress import serve

#%% Load Model from pickle
model = pickle.load(open('models/house_quotes.sav', 'rb'))

# Columns for X table
columns = ['tamanho', 'ano', 'garagem']
#%% Application
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app) # Needed on each end point

# Defining routes
@app.route('/')
def home():
    return "My first API."

    # End point to the url
@app.route('/sentiment/<message>') #"message" is object here, it's passed in the url not in the page
@basic_auth.required
def sentiment(message):
    translator = Translator()
    #translation = translator.translate(message, src='pt', dest='en') # Use of 'src' can limitate the translations
    translation = translator.translate(message, dest='en')
    tb_en = TextBlob(translation.text)
    polarity = tb_en.polarity # Blob range from -1 to 1.
    return 'Polarity: {}'.format(polarity)

    # End point to the ML model (default methods = get)
@app.route('/quotes/<int:size>') # The size needs to be integer
def quote(size):
    price = model.predict([[size]])
    return str(price)

    # End point to PAYLOAD
@app.route('/quotes_list/', methods=['POST'])
@basic_auth.required
def quote_lsit():
    data = request.get_json()
    input_data = [data[col] for col in columns] # to be sure the data will follow the model's order
    price = model.predict([input_data])
    return jsonify(price=price[0]) # Use 0 to get the first value from the np.array


app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT')) # to do the deploy in more than one environment. (Docker, local, etc)
# %%
'''
Note: Some modifications were necessary to run the app in windows. Because
of that, there's an error in professor's version that he fix with:

if __name__ == __main__:
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT'))

He calls the app.run because he is using gunicorn (doesn't work on Windows).

Since my code uses waitress, the following code is enough to run without errors
until this point.

PS: Course link: https://cursos.alura.com.br/course/mlops-deploy-modelos/task/83754
    Class 03 - Aplicação servless com App Engine
    Topic 03 - Arquivo de configuração
'''
serve(app.run, listen='*:'+os.environ.get('PORT'))