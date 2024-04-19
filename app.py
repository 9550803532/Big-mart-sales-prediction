from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import os
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  
registered_users = {
    'admin': 'password',
    'user1': 'password1',
    'user2': 'password2',
    # Add more users as needed
}


@app.route("/")
def index():
    if 'username' in session:
        return redirect(url_for('home'))
    else:
        return redirect(url_for('login'))

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in registered_users and registered_users[username] == password:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return "Invalid username or password"
    
    return render_template('login.html')

@app.route("/logout", methods=['POST'])
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route("/home")
def home():
    if 'username' in session:
        return render_template("home.html")
    else:
        return redirect(url_for('login'))


@app.route('/predict', methods=['POST'])
def result():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    item_weight = float(request.form['item_weight'])
    item_fat_content = float(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = float(request.form['outlet_establishment_year'])
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])

    X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                   outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

    scaler_path = r'C:\Users\Ramya Sri\Downloads\BigMart-Sales-Prediction-With-Deployment-main (1)\BigMart-Sales-Prediction-With-Deployment-main\models\sc.sav'

    sc = joblib.load(scaler_path)

    X_std = sc.transform(X)

    model_path = r'C:\Users\Ramya Sri\Downloads\BigMart-Sales-Prediction-With-Deployment-main (1)\BigMart-Sales-Prediction-With-Deployment-main\models\lr.sav'

    model = joblib.load(model_path)

    Y_pred = model.predict(X_std)

    return render_template("result.html", prediction=float(Y_pred))


if __name__ == "__main__":
    app.run(debug=True, port=9457)
