from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pickle

# Load ML model
model = pickle.load(open('model.pkl', 'rb'))

# Create application
app = Flask(__name__)


@app.route('/')  # ROOT
def index():
    result_message = request.args.get('result', None)
    prediction = request.args.get('prediction', None)
    return render_template('web.html', result=result_message, prediction=prediction)


@app.route('/predict', methods=['POST'])  # POST
def predict():
    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    # # Convert features to array
    array_features = [np.array(features)]
    # # Predict features
    prediction = model.predict(array_features)

    # Determine the result message
    result_message = 'more chance of heart attack' if prediction == 1 else 'less chance of heart attack'

    # Redirect to the homepage with the result message
    return redirect(url_for('index', result=result_message, prediction=prediction[0]))


if __name__ == '__main__':
    app.run(debug=True)
