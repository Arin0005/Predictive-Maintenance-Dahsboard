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
    MISALIGN_MODEL_PATH = r'misalign/misalign_predictive_maintenance_model.h5'
    MISALIGN_SCALER_X_PATH = r'misalign/scaler_X.pkl'
    MISALIGN_SCALER_Y_PATH = r'misalign/scaler_y.pkl'
    
    # BPFI Model
    BPFI_MODEL_PATH = r'BPFI/predictive_maintenance_model.h5'
    BPFI_SCALER_X_PATH = r'BPFI/scaler_X.pkl'
    BPFI_SCALER_Y_PATH = r'BPFI/scaler_y.pkl'

config = Config()

# Load models and scalers at startup
print("Loading models and scalers...")
models = {}
scalers = {}

try:
    # Load Misalignment Model
    models['misalign'] = keras.models.load_model(config.MISALIGN_MODEL_PATH, compile=False)
    scalers['misalign_X'] = joblib.load(config.MISALIGN_SCALER_X_PATH)
    scalers['misalign_y'] = joblib.load(config.MISALIGN_SCALER_Y_PATH)
    print("Misalignment model loaded successfully!")
except Exception as e:
    print(f"Error loading misalignment model: {e}")
    models['misalign'] = None

try:
    # Load BPFI Model
    models['bpfi'] = keras.models.load_model(config.BPFI_MODEL_PATH, compile=False)
    scalers['bpfi_X'] = joblib.load(config.BPFI_SCALER_X_PATH)
    scalers['bpfi_y'] = joblib.load(config.BPFI_SCALER_Y_PATH)
    print("BPFI model loaded successfully!")
except Exception as e:
    print(f"Error loading BPFI model: {e}")
    models['bpfi'] = None

# ==================== HELPER FUNCTIONS ====================

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
    df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')
    
    return df_resampled

def extract_time_domain_features(window):
    """Extract time-domain features"""
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
    """Extract frequency-domain features"""
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
    """Create windowed features from sensor data"""
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
        
        # Temperature features
        for col in df_temp.columns:
            temp_window = df_temp[col].values[start_idx:end_idx]
            time_feats = extract_time_domain_features(temp_window)
            for feat_name, feat_val in time_feats.items():
                window_features[f'temp_{col}_{feat_name}'] = feat_val
        
        # Vibration features
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
    """Complete pipeline: preprocess, extract features, predict"""
    
    # Select appropriate model and scalers
    model = models.get(model_type)
    scaler_X = scalers.get(f'{model_type}_X')
    scaler_y = scalers.get(f'{model_type}_y')
    
    if model is None or scaler_X is None or scaler_y is None:
        raise ValueError(f"Model '{model_type}' not loaded properly")
    
    # Step 1: Preprocess
    df_temp_resampled = preprocess_dataframe(df_temp)
    df_vib_resampled = preprocess_dataframe(df_vib)
    
    # Step 2: Extract features
    feature_df = create_windowed_features(df_temp_resampled, df_vib_resampled)
    
    # Step 3: Create sequences
    X = feature_df.values
    X_sequences = []
    for i in range(len(X) - config.SEQUENCE_LENGTH + 1):
        X_sequences.append(X[i:i+config.SEQUENCE_LENGTH])
    X_sequences = np.array(X_sequences)
    
    # Step 4: Scale features
    X_reshaped = X_sequences.reshape(-1, X_sequences.shape[-1])
    print(f"Feature shape before scaling: {X_reshaped.shape}")
    X_scaled = scaler_X.transform(X_reshaped)
    X_test_scaled = X_scaled.reshape(X_sequences.shape)
    print(f"Feature shape after scaling: {X_test_scaled.shape}")
    
    # Step 5: Predict
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    return y_pred, feature_df

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

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
    """Handle prediction request"""
    try:
        # Get model type
        model_type = request.form.get('model_type', 'misalign')
        
        # Check if files are present
        if 'temp_file' not in request.files or 'vib_file' not in request.files:
            return jsonify({'error': 'Both temperature and vibration files are required'}), 400
        
        temp_file = request.files['temp_file']
        vib_file = request.files['vib_file']
        
        # Check if files are empty
        if temp_file.filename == '' or vib_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV files
        df_temp = pd.read_csv(temp_file)
        df_vib = pd.read_csv(vib_file)
        
        print(f"Loaded temp data: {df_temp.shape}")
        print(f"Loaded vib data: {df_vib.shape}")
        print(f"Model type: {model_type}")
        
        # Process and predict
        y_pred, feature_df = process_and_predict(df_temp, df_vib, model_type)
        
        # Create results
        sequence_indices = np.arange(len(y_pred))
        rul_minutes = y_pred.tolist()
        rul_hours = (y_pred / 60).tolist()
        
        health_states = []
        maintenance_urgency = []
        for rul in y_pred:
            health_state, urgency = classify_health_state(rul)
            health_states.append(health_state)
            maintenance_urgency.append(urgency)
        
        # Calculate comprehensive statistics
        stats_data = {
            'total_sequences': int(len(y_pred)),
            'mean_rul_minutes': float(np.mean(y_pred)),
            'median_rul_minutes': float(np.median(y_pred)),
            'mean_rul_hours': float(np.mean(y_pred) / 60),
            'min_rul_minutes': float(np.min(y_pred)),
            'max_rul_minutes': float(np.max(y_pred)),
            # 'std_rul_minutes': float(np.std(y_pred)),
            # 'variance_rul_minutes': float(np.var(y_pred)),
            # 'range_rul_minutes': float(np.ptp(y_pred)),
            # 'q1_rul_minutes': float(np.percentile(y_pred, 25)),
            # 'q3_rul_minutes': float(np.percentile(y_pred, 75)),
            # 'iqr_rul_minutes': float(np.percentile(y_pred, 75) - np.percentile(y_pred, 25)),
            'healthy_count': int(sum(1 for s in health_states if s == 'Healthy')),
            'warning_count': int(sum(1 for s in health_states if s == 'Warning')),
            'severe_count': int(sum(1 for s in health_states if s == 'Severe')),
            'critical_count': int(sum(1 for s in health_states if s == 'Critical')),
            'model_type': model_type
        }
        
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
            'message': f'Predictions generated successfully using {model_type} model'
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during prediction'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)