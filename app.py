#app.py

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy import signal, stats
import joblib
import tensorflow as tf
from tensorflow import keras
import os
import traceback
from pathlib import Path

# Import classification module
from classify import classify_bearing

app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    ORIGINAL_FREQ = 25600
    TARGET_FREQ = 1000
    WINDOW_SIZE = 2000
    OVERLAP = 0.5
    SEQUENCE_LENGTH = 10
    
    # Misalignment Model
    MISALIGN_MODEL_PATH = Path('misalign')/"misalign_predictive_maintenance_model.h5"
    MISALIGN_SCALER_X_PATH = Path('misalign')/'scaler_X.pkl'
    MISALIGN_SCALER_Y_PATH = Path('misalign')/'scaler_y.pkl'

    # BPFI Model
    BPFI_MODEL_PATH = Path('BPFI')/'predictive_maintenance_model.h5'
    BPFI_SCALER_X_PATH = Path('BPFI')/'scaler_X.pkl'
    BPFI_SCALER_Y_PATH = Path('BPFI')/'scaler_y.pkl'

config = Config()

# Load RUL prediction models and scalers at startup
print("Loading RUL prediction models and scalers...")
models = {}
scalers = {}

try:
    # Load Misalignment Model
    models['misalign'] = keras.models.load_model(config.MISALIGN_MODEL_PATH, compile=False)
    scalers['misalign_X'] = joblib.load(config.MISALIGN_SCALER_X_PATH)
    scalers['misalign_y'] = joblib.load(config.MISALIGN_SCALER_Y_PATH)
    print(f"  Misalignment model loaded (expects {scalers['misalign_X'].n_features_in_} features)")
except Exception as e:
    print(f"   Error loading misalignment model: {e}")
    models['misalign'] = None

try:
    # Load BPFI Model
    models['bpfi'] = keras.models.load_model(config.BPFI_MODEL_PATH, compile=False)
    scalers['bpfi_X'] = joblib.load(config.BPFI_SCALER_X_PATH)
    scalers['bpfi_y'] = joblib.load(config.BPFI_SCALER_Y_PATH)
    print(f"  BPFI model loaded (expects {scalers['bpfi_X'].n_features_in_} features)")
except Exception as e:
    print(f"   Error loading BPFI model: {e}")
    models['bpfi'] = None

# ==================== HELPER FUNCTIONS FOR RUL PREDICTION ====================

def resample_signal(signal_data, original_freq, target_freq):
    """Resample signal from original frequency to target frequency"""
    num_samples = len(signal_data)
    duration = num_samples / original_freq
    target_samples = int(duration * target_freq)
    resampled = signal.resample(signal_data, target_samples)
    return resampled

def preprocess_dataframe(df):
    """Preprocess dataframe: remove timestamp, resample, clean"""
    df_processed = df.copy()
    
    if 'timestamp' in df_processed.columns:
        df_processed = df_processed.drop('timestamp', axis=1)
    
    resampled_data = {}
    for col in df_processed.columns:
        resampled_data[col] = resample_signal(
            df_processed[col].values,
            config.ORIGINAL_FREQ,
            config.TARGET_FREQ
        )
    
    df_resampled = pd.DataFrame(resampled_data)
    df_resampled = df_resampled.replace([np.inf, -np.inf], np.nan)
    # Fix deprecation warning
    df_resampled = df_resampled.ffill().bfill()
    
    return df_resampled

def extract_time_domain_features(window):
    """Extract time-domain features for RUL prediction"""
    features = {}
    features['mean'] = np.mean(window)
    features['std'] = np.std(window)
    features['rms'] = np.sqrt(np.mean(window**2))
    features['peak'] = np.max(np.abs(window))
    features['peak_to_peak'] = np.ptp(window)
    features['crest_factor'] = features['peak'] / (features['rms'] + 1e-8)
    features['kurtosis'] = stats.kurtosis(window)
    features['skewness'] = stats.skew(window)
    return features

def extract_frequency_domain_features(window, fs=1000):
    """Extract frequency-domain features for RUL prediction"""
    features = {}
    
    fft_vals = np.fft.fft(window)
    fft_freq = np.fft.fftfreq(len(window), 1/fs)
    fft_power = np.abs(fft_vals)**2
    
    pos_mask = fft_freq > 0
    fft_freq_pos = fft_freq[pos_mask]
    fft_power_pos = fft_power[pos_mask]
    
    features['spectral_centroid'] = np.sum(fft_freq_pos * fft_power_pos) / (np.sum(fft_power_pos) + 1e-8)
    features['spectral_variance'] = np.sqrt(np.sum(((fft_freq_pos - features['spectral_centroid'])**2) * fft_power_pos) / (np.sum(fft_power_pos) + 1e-8))
    
    misalign_range = (fft_freq_pos >= 100) & (fft_freq_pos <= 200)
    features['misalign_energy'] = np.sum(fft_power_pos[misalign_range])
    
    return features

def create_windowed_features(df_temp, df_vib):
    """Create windowed features from sensor data for RUL prediction"""
    window_size = config.WINDOW_SIZE
    overlap = config.OVERLAP
    step_size = int(window_size * (1 - overlap))
    num_windows = (len(df_temp) - window_size) // step_size + 1
    
    features_list = []
    
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        if end_idx > len(df_temp):
            break
        
        window_features = {}
        
        # Temperature features (time-domain only for RUL)
        for col in df_temp.columns:
            temp_window = df_temp[col].values[start_idx:end_idx]
            time_feats = extract_time_domain_features(temp_window)
            for feat_name, feat_val in time_feats.items():
                window_features[f'temp_{col}_{feat_name}'] = feat_val
        
        # Vibration features (time + freq domain for RUL)
        for col in df_vib.columns:
            vib_window = df_vib[col].values[start_idx:end_idx]
            time_feats = extract_time_domain_features(vib_window)
            freq_feats = extract_frequency_domain_features(vib_window)
            for feat_name, feat_val in {**time_feats, **freq_feats}.items():
                window_features[f'vib_{col}_{feat_name}'] = feat_val
        
        features_list.append(window_features)
    
    return pd.DataFrame(features_list)

def classify_health_state(rul):
    """Classify health state based on RUL with 4 categories"""
    if rul > 60:
        return 'Healthy', 'Low'
    elif rul > 40:
        return 'Warning', 'Medium'
    elif rul > 20:
        return 'Severe', 'High'
    else:
        return 'Critical', 'Critical'

def process_and_predict(df_temp, df_vib, model_type='misalign'):
    """Complete pipeline: preprocess, extract features, predict RUL"""
    
    # Select appropriate model and scalers
    model = models.get(model_type)
    scaler_X = scalers.get(f'{model_type}_X')
    scaler_y = scalers.get(f'{model_type}_y')
    
    if model is None or scaler_X is None or scaler_y is None:
        raise ValueError(f"Model '{model_type}' not loaded properly")
    
    print(f"\n{'='*60}")
    print(f"RUL PREDICTION PIPELINE ({model_type.upper()})")
    print(f"{'='*60}")
    
    # Step 1: Preprocess
    print("Step 1: Preprocessing data...")
    df_temp_resampled = preprocess_dataframe(df_temp)
    df_vib_resampled = preprocess_dataframe(df_vib)
    print(f"  Temp shape after resample: {df_temp_resampled.shape}")
    print(f"  Vib shape after resample: {df_vib_resampled.shape}")
    
    # Step 2: Extract features
    print("\nStep 2: Extracting features...")
    feature_df = create_windowed_features(df_temp_resampled, df_vib_resampled)
    print(f"  Feature DataFrame shape: {feature_df.shape}")
    print(f"  Number of features extracted: {feature_df.shape[1]}")
    print(f"  Model expects: {scaler_X.n_features_in_} features")
    
    # Check feature count match
    if feature_df.shape[1] != scaler_X.n_features_in_:
        print(f"\n  WARNING: Feature mismatch!")
        print(f"  Extracted: {feature_df.shape[1]} features")
        print(f"  Expected: {scaler_X.n_features_in_} features")
        print(f"  Difference: {feature_df.shape[1] - scaler_X.n_features_in_}")
        
        # Try to fix by selecting or padding features
        if feature_df.shape[1] > scaler_X.n_features_in_:
            print(f"  → Trimming to first {scaler_X.n_features_in_} features")
            feature_df = feature_df.iloc[:, :scaler_X.n_features_in_]
        else:
            print(f"  → Padding with zeros to {scaler_X.n_features_in_} features")
            padding = pd.DataFrame(
                np.zeros((feature_df.shape[0], scaler_X.n_features_in_ - feature_df.shape[1])),
                columns=[f'pad_{i}' for i in range(scaler_X.n_features_in_ - feature_df.shape[1])]
            )
            feature_df = pd.concat([feature_df, padding], axis=1)
    
    # Step 3: Create sequences
    print("\nStep 3: Creating sequences...")
    X = feature_df.values
    X_sequences = []
    for i in range(len(X) - config.SEQUENCE_LENGTH + 1):
        X_sequences.append(X[i:i+config.SEQUENCE_LENGTH])
    X_sequences = np.array(X_sequences)
    print(f"  Sequence shape: {X_sequences.shape}")
    
    # Step 4: Scale features
    print("\nStep 4: Scaling features...")
    X_reshaped = X_sequences.reshape(-1, X_sequences.shape[-1])
    X_scaled = scaler_X.transform(X_reshaped)
    X_test_scaled = X_scaled.reshape(X_sequences.shape)
    print(f"  Scaled shape: {X_test_scaled.shape}")
    
    # Step 5: Predict
    print("\nStep 5: Predicting RUL...")
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    print(f"  Predictions generated: {len(y_pred)} sequences")
    print(f"  RUL range: {y_pred.min():.1f} - {y_pred.max():.1f} minutes")
    print(f"{'='*60}\n")
    
    return y_pred, feature_df

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Serve the home page"""
    return render_template('index.html')

@app.route('/trymodel')
def trymodel():
    """Serve the try model/dashboard page"""
    return render_template('trymodel.html')

@app.route('/faq')
def faq():
    """Serve the FAQ page"""
    return render_template('faq.html')

@app.route('/about-us')
def about_us():
    """Serve the About Us page"""
    return render_template('about-us.html')

@app.route('/about-modal')
def about_modal():
    """Serve the About Model page"""
    return render_template('about-modal.html')

@app.route('/modal-evaluation')
def modal_evaluation():
    """Serve the Model Evaluation page"""
    return render_template('modal-evaluation.html')

@app.route('/data-collection')
def data_collection():
    """Serve the Data Collection page"""
    return render_template('data-collection.html')

@app.route('/maintenance-evolution')
def maintenance_evolution():
    """Serve the Maintenance Evolution page"""
    return render_template('maintenance-evolution.html')

@app.route('/future-scope')
def future_scope():
    """Serve the Future Scope page"""
    return render_template('future-scope.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Check if models are loaded"""
    misalign_loaded = models.get('misalign') is not None
    bpfi_loaded = models.get('bpfi') is not None
    
    return jsonify({
        'status': 'healthy' if (misalign_loaded or bpfi_loaded) else 'error',
        'models': {
            'misalign': misalign_loaded,
            'bpfi': bpfi_loaded
        },
        'message': 'System ready for predictions'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle RUL prediction and classification request"""
    try:
        # Get model type
        model_type = request.form.get('model_type', 'misalign')
        
        # Check if files are present
        if 'temp_file' not in request.files or 'vib_file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Both temperature and vibration files are required'
            }), 400
        
        temp_file = request.files['temp_file']
        vib_file = request.files['vib_file']
        
        # Check if files are empty
        if temp_file.filename == '' or vib_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Read CSV files
        df_temp = pd.read_csv(temp_file)
        df_vib = pd.read_csv(vib_file)
        
        print(f"\n{'='*60}")
        print(f"NEW PREDICTION REQUEST")
        print(f"{'='*60}")
        print(f"Temp data shape: {df_temp.shape}")
        print(f"Vib data shape: {df_vib.shape}")
        print(f"Model type: {model_type}")
        
        # Process and predict RUL
        y_pred, feature_df = process_and_predict(df_temp, df_vib, model_type)
        
        # Create RUL results
        sequence_indices = np.arange(len(y_pred))
        rul_minutes = y_pred.tolist()
        rul_hours = (y_pred / 60).tolist()
        
        health_states = []
        maintenance_urgency = []
        for rul in y_pred:
            health_state, urgency = classify_health_state(rul)
            health_states.append(health_state)
            maintenance_urgency.append(urgency)
        
        # Count each category
        healthy_count = int(sum(1 for s in health_states if s == 'Healthy'))
        warning_count = int(sum(1 for s in health_states if s == 'Warning'))
        severe_count = int(sum(1 for s in health_states if s == 'Severe'))
        critical_count = int(sum(1 for s in health_states if s == 'Critical'))
        
        # Calculate comprehensive statistics
        stats_data = {
            'total_sequences': int(len(y_pred)),
            'mean_rul_minutes': float(np.mean(y_pred)),
            'median_rul_minutes': float(np.median(y_pred)),
            'mean_rul_hours': float(np.mean(y_pred) / 60),
            'min_rul_minutes': float(np.min(y_pred)),
            'max_rul_minutes': float(np.max(y_pred)),
            'std_rul_minutes': float(np.std(y_pred)),
            'variance_rul_minutes': float(np.var(y_pred)),
            'range_rul_minutes': float(np.ptp(y_pred)),
            'q1_rul_minutes': float(np.percentile(y_pred, 25)),
            'q3_rul_minutes': float(np.percentile(y_pred, 75)),
            'iqr_rul_minutes': float(np.percentile(y_pred, 75) - np.percentile(y_pred, 25)),
            'healthy_count': healthy_count,
            'warning_count': warning_count,
            'severe_count': severe_count,
            'critical_count': critical_count,
            'model_type': model_type
        }
        
        # Run classification on the same data
        print(f"\n{'='*60}")
        print("RUNNING CLASSIFICATION...")
        print(f"{'='*60}")
        
        classification_results = {}
        try:
            # Reset file pointers and read again for classification
            temp_file.seek(0)
            vib_file.seek(0)
            df_temp_classify = pd.read_csv(temp_file)
            df_vib_classify = pd.read_csv(vib_file)
            
            classification_results = classify_bearing(df_temp_classify, df_vib_classify)
            
            if classification_results:
                print(f"  Classification completed successfully!")
                print(f"  Models used: {', '.join(classification_results.keys())}")
            else:
                print("  Classification returned no results")
                
        except Exception as e:
            print(f"   Classification failed: {str(e)}")
            print(traceback.format_exc())
            # Don't fail the entire request if classification fails
            classification_results = {}
        
        # Prepare response
        response = {
            'success': True,
            'predictions': {
                'sequence_indices': sequence_indices.tolist(),
                'rul_minutes': rul_minutes,
                'rul_hours': rul_hours,
                'health_states': health_states,
                'maintenance_urgency': maintenance_urgency
            },
            'statistics': stats_data,
            'classifications': classification_results,
            'message': f'Predictions generated successfully using {model_type} model'
        }
        
        print(f"\n{'='*60}")
        print("REQUEST COMPLETED SUCCESSFULLY")
        print(f"{'='*60}\n")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR DURING PREDICTION")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during prediction'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)