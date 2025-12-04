from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pickle
import os

app = Flask(__name__)

MODEL_ACCURACY = 99.32

# -------------------- DATABASE CONFIGURATION --------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crop_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Farmer and CropRecommendation Tables
class Farmer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class CropRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    farmer_email = db.Column(db.String(120), nullable=False)
    crop_name = db.Column(db.String(100), nullable=False)
    N = db.Column(db.Float)
    P = db.Column(db.Float)
    K = db.Column(db.Float)
    Temperature = db.Column(db.Float)
    Humidity = db.Column(db.Float)
    PH = db.Column(db.Float)
    Rainfall = db.Column(db.Float)

# Create database tables
with app.app_context():
    db.create_all()

# -------------------- MODEL LOADING --------------------
MODEL_PATH = 'crop_recommendation_model.pkl'
model = None
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        print("⚠️ Model file not found. Using mock prediction.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Mock prediction if model not available
def predict_crop_mock(features):
    N, P, K, temp, humidity, ph, rainfall = features
    if N > 100 and temp > 25:
        return "Cotton"
    elif K > 50 and ph < 6:
        return "Rice"
    elif P > 50 and humidity < 70:
        return "Wheat"
    elif temp < 20 and rainfall > 200:
        return "Grapes"
    else:
        return "Maize"

# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return render_template('index.html')

# Define the Route for the About Us page
@app.route('/about')  # The URL path the user visits (e.g., 127.0.0.1:5000/about)
def about():
    # Tells the server to find and render the 'about.html' file from the 'templates' folder
    return render_template('about.html')

# Define the Route for the Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if Farmer.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already registered!'}), 400

    new_farmer = Farmer(email=email, password=password)
    db.session.add(new_farmer)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Registration successful!'})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    farmer = Farmer.query.filter_by(email=email, password=password).first()
    if farmer:
        return jsonify({'success': True, 'message': f'Welcome {email}!'})
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials.'}), 401

@app.route('/crop_recommendation', methods=['POST'])
def crop_recommendation():
    """
    Handles crop recommendation request, runs ML prediction, and saves the data.
    """
    if model is None:
        return jsonify({'success': False, 'message': 'ML Model is not loaded.'}), 503

    try:
        data = request.json
        # 1. Extract and validate input data
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        Temp = float(data['Temperature'])
        Hum = float(data['Humidity'])
        PH = float(data['PH'])
        Rain = float(data['Rainfall'])
        farmer_email = data['farmer_email'] # Assuming email is passed for saving

        # 2. Prepare data for model
        features = np.array([[N, P, K, Temp, Hum, PH, Rain]])

        # 3. Predict crop
        prediction = model.predict(features)[0]

        # 4. Save recommendation to database
        new_rec = CropRecommendation(
            farmer_email=farmer_email,
            crop_name=prediction,
            N=N, P=P, K=K,
            Temperature=Temp, Humidity=Hum, PH=PH, Rainfall=Rain
        )
        with app.app_context():
            db.session.add(new_rec)
            db.session.commit()

        # 5. Return prediction and MODEL_ACCURACY
        return jsonify({
            'success': True,
            'recommended_crop': prediction,
            'model_accuracy': MODEL_ACCURACY, # NEW: Include the model's overall accuracy
            'message': f'Crop recommended successfully!'
        })
    except Exception as e:
        print(f"Error: {e}")
        # Log the full error for debugging but return a simpler message to the user
        return jsonify({'success': False, 'message': 'Error processing request. Check input data.'}), 500
    
@app.route('/admin_data', methods=['GET'])
def admin_data():
    try:
        # Fetch all farmers
        farmers = Farmer.query.all()
        farmer_list = [{'id': f.id, 'email': f.email} for f in farmers]

        # Fetch all recommendations
        recs = CropRecommendation.query.order_by(CropRecommendation.id.desc()).all()
        rec_list = []
        for r in recs:
            rec_list.append({
                'id': r.id,
                'farmer_email': r.farmer_email,
                'crop_name': r.crop_name,
                'N': r.N,
                'P': r.P,
                'K': r.K,
                'Temperature': r.Temperature,
                'Humidity': r.Humidity,
                'PH': r.PH,
                'Rainfall': r.Rainfall
            })

        return jsonify({
            'success': True,
            'farmers': farmer_list,
            'recommendations': rec_list
        })
    except Exception as e:
        print(f"Admin Data Error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
