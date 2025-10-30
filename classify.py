"""
classify.py - Bearing Fault Classification Module
Fixed version with correct feature extraction matching trained models (114 features)
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew, entropy
from scipy.fft import fft, fftfreq
import joblib
from tensorflow import keras
import os
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

class BearingClassifier:
    """Main classifier for bearing fault detection"""
    
    def __init__(self):
        """Initialize classifier with hardcoded model paths"""
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load all models with hardcoded paths and error handling"""
        try:
            # MANUAL PATHS - UPDATE THESE TO YOUR ACTUAL FILE PATHS
            FEATURE_SCALER_PATH = r'classification\feature_scaler.pkl'
            LABEL_ENCODER_PATH = r'classification\label_encoder.pkl'
            FEATURE_COLUMNS_PATH = r'classification\feature_columns.pkl'
            RANDOM_FOREST_PATH = r'classification\random_forest_model.pkl'
            XGBOOST_PATH = r'D:\Github\Predictive-Maintenance-Dahsboard\classification\random_forest_model.pkl'
            DNN_PATH = r'classification\dnn_model.h5'
            
            print("\n" + "="*60)
            print("LOADING CLASSIFICATION MODELS")
            print("="*60)
            
            # Load preprocessing objects
            print("Loading preprocessing objects...")
            self.scalers['feature_scaler'] = joblib.load(FEATURE_SCALER_PATH)
            print(f"    Feature scaler loaded (expects {self.scalers['feature_scaler'].n_features_in_} features)")
            
            self.scalers['label_encoder'] = joblib.load(LABEL_ENCODER_PATH)
            print(f"    Label encoder loaded ({len(self.scalers['label_encoder'].classes_)} classes)")
            
            self.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
            print(f"    Feature columns loaded ({len(self.feature_columns)} features)")
            
            # Try loading Random Forest
            try:
                print("\nLoading Random Forest model...")
                self.models['random_forest'] = joblib.load(RANDOM_FOREST_PATH)
                print("    Random Forest loaded")
            except Exception as e:
                print(f"    Random Forest failed: {str(e)[:100]}")
                self.models['random_forest'] = None
            
            # Try loading XGBoost
            try:
                print("Loading XGBoost model...")
                self.models['xgboost'] = joblib.load(XGBOOST_PATH)
                print("    XGBoost loaded")
            except Exception as e:
                print(f"    XGBoost failed: {str(e)[:100]}")
                self.models['xgboost'] = None
            
            # Try loading DNN
            try:
                print("Loading DNN model...")
                self.models['dnn'] = keras.models.load_model(DNN_PATH, compile=False)
                print("    DNN loaded")
            except Exception as e:
                print(f"    DNN failed: {str(e)[:100]}")
                self.models['dnn'] = None
            
            # Check if at least one model loaded
            loaded_models = [k for k, v in self.models.items() if v is not None]
            
            if len(loaded_models) > 0:
                print(f"\n  Successfully loaded {len(loaded_models)} model(s): {', '.join(loaded_models)}")
                self.models_loaded = True
                print("="*60 + "\n")
                return True
            else:
                print("\n  No models loaded successfully!")
                print("="*60 + "\n")
                self.models_loaded = False
                return False
            
        except Exception as e:
            print(f"\n  Critical error loading classification models: {e}")
            print("\nPlease check that these files exist:")
            print(f"  Feature Scaler: {FEATURE_SCALER_PATH}")
            print(f"  Label Encoder: {LABEL_ENCODER_PATH}")
            print(f"  Feature Columns: {FEATURE_COLUMNS_PATH}")
            print("="*60 + "\n")
            self.models_loaded = False
            return False
    
    def downsample_signal(self, signal_data, original_fs=25600, target_fs=1000):
        """
        Downsample signal from original_fs to target_fs
        Uses anti-aliasing filter to prevent aliasing
        """
        downsample_factor = int(original_fs / target_fs)
        
        # Apply anti-aliasing filter
        nyquist = target_fs / 2
        cutoff = nyquist * 0.8  # 80% of Nyquist
        
        # Design low-pass Butterworth filter
        sos = signal.butter(8, cutoff, btype='low', fs=original_fs, output='sos')
        filtered = signal.sosfilt(sos, signal_data)
        
        # Downsample
        downsampled = filtered[::downsample_factor]
        
        return downsampled
    
    def segment_signal(self, signal_data, segment_length=1000, overlap=0.5):
        """
        Segment signal into overlapping windows
        segment_length: samples per segment (e.g., 1000 = 1 sec at 1000 Hz)
        overlap: fraction of overlap (0.5 = 50% overlap)
        """
        step = int(segment_length * (1 - overlap))
        segments = []
        
        for i in range(0, len(signal_data) - segment_length + 1, step):
            segments.append(signal_data[i:i + segment_length])
        
        return np.array(segments)
    
    def extract_time_domain_features(self, signal_data):
        """
        Extract time-domain statistical features
        Must match the features used during training (114 total)
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        features['peak'] = np.max(np.abs(signal_data))
        features['peak_to_peak'] = np.ptp(signal_data)
        features['crest_factor'] = features['peak'] / (features['rms'] + 1e-10)

        # Higher order statistics
        features['kurtosis'] = kurtosis(signal_data)
        features['skewness'] = skew(signal_data)
        features['shape_factor'] = features['rms'] / (np.mean(np.abs(signal_data)) + 1e-10)
        features['impulse_factor'] = features['peak'] / (np.mean(np.abs(signal_data)) + 1e-10)
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
        
        return features

    def extract_frequency_domain_features(self, signal_data, fs=1000):
        """
        Extract frequency-domain features using FFT
        Must match the features used during training (114 total)
        """
        features = {}
        
        # Compute FFT
        n = len(signal_data)
        fft_vals = fft(signal_data)
        fft_freq = fftfreq(n, 1/fs)
        
        # Only positive frequencies
        positive_freq_idx = fft_freq > 0
        fft_mag = np.abs(fft_vals[positive_freq_idx])
        fft_freq = fft_freq[positive_freq_idx]
        
        # Normalize
        fft_mag = fft_mag / n
        
        # Spectral features
        features['spectral_centroid'] = np.sum(fft_freq * fft_mag) / (np.sum(fft_mag) + 1e-10)
        features['spectral_spread'] = np.sqrt(np.sum(((fft_freq - features['spectral_centroid'])**2) * fft_mag) / (np.sum(fft_mag) + 1e-10))
        features['spectral_skewness'] = np.sum(((fft_freq - features['spectral_centroid'])**3) * fft_mag) / ((features['spectral_spread']**3) * np.sum(fft_mag) + 1e-10)
        features['spectral_kurtosis'] = np.sum(((fft_freq - features['spectral_centroid'])**4) * fft_mag) / ((features['spectral_spread']**4) * np.sum(fft_mag) + 1e-10)
        
        # Spectral entropy
        psd = fft_mag**2
        psd_norm = psd / (np.sum(psd) + 1e-10)
        features['spectral_entropy'] = entropy(psd_norm + 1e-10)
        
        # Dominant frequency
        features['dominant_frequency'] = fft_freq[np.argmax(fft_mag)]
        features['max_magnitude'] = np.max(fft_mag)
        
        # Band powers (divide spectrum into 4 bands)
        freq_bands = [(0, 50), (50, 150), (150, 300), (300, 500)]
        for i, (low, high) in enumerate(freq_bands):
            band_idx = (fft_freq >= low) & (fft_freq < high)
            features[f'band_power_{i}'] = np.sum(fft_mag[band_idx]**2)
        
        return features

    def extract_features_from_signal(self, signal_data, fs=1000):
        """
        Extract all features from a signal
        Returns dict with time_ and freq_ prefixed features
        """
        features = {}
        
        # Time domain features
        time_features = self.extract_time_domain_features(signal_data)
        features.update({f'time_{k}': v for k, v in time_features.items()})
        
        # Frequency domain features
        freq_features = self.extract_frequency_domain_features(signal_data, fs)
        features.update({f'freq_{k}': v for k, v in freq_features.items()})
        
        return features
    
    def get_maintenance_recommendation(self, fault_type, severity):
        """Generate maintenance recommendation based on fault type and severity"""
        if fault_type == 'Normal':
            return 'No maintenance required - System operating normally', 'Normal Operation'
        
        # Determine fault description and recommendation
        if 'BPFI' in fault_type:
            fault_desc = 'Bearing Inner Race Fault'
            if '0.3mm' in severity:
                maintenance = 'CAUTION: Minor inner race fault detected. Schedule inspection within 1 week.'
            elif '1.0mm' in severity:
                maintenance = 'WARNING: Moderate inner race fault. Schedule bearing replacement within 2-3 days.'
            else:
                maintenance = 'CRITICAL: Severe inner race fault. Immediate bearing replacement required!'
                
        elif 'BPFO' in fault_type:
            fault_desc = 'Bearing Outer Race Fault'
            if '0.3mm' in severity:
                maintenance = 'CAUTION: Minor outer race fault detected. Schedule inspection within 1 week.'
            elif '1.0mm' in severity:
                maintenance = 'WARNING: Moderate outer race fault. Schedule bearing replacement within 2-3 days.'
            else:
                maintenance = 'CRITICAL: Severe outer race fault. Immediate bearing replacement required!'
                
        elif 'Misalign' in fault_type:
            fault_desc = 'Shaft Misalignment'
            if '0.1mm' in severity:
                maintenance = 'CAUTION: Minor misalignment detected. Schedule alignment check within 1 week.'
            elif '0.3mm' in severity:
                maintenance = 'WARNING: Moderate misalignment. Perform shaft alignment within 2-3 days.'
            else:
                maintenance = 'CRITICAL: Severe misalignment. Immediate shaft alignment required!'
                
        elif 'Unbalance' in fault_type:
            fault_desc = 'Rotor Unbalance'
            if '583mg' in severity or '1169mg' in severity:
                maintenance = 'CAUTION: Minor unbalance detected. Schedule balancing within 1 week.'
            elif '1751mg' in severity or '2239mg' in severity:
                maintenance = 'WARNING: Moderate unbalance. Perform rotor balancing within 2-3 days.'
            else:
                maintenance = 'CRITICAL: Severe unbalance. Immediate rotor balancing required!'
        else:
            fault_desc = fault_type
            maintenance = 'Inspect and perform necessary maintenance.'
        
        return maintenance, fault_desc
    
    def predict_single_model(self, feature_vector, model_name):
        """Predict using a single model"""
        model = self.models.get(model_name)
        
        if model is None:
            return None, None
        
        try:
            if model_name == 'dnn':
                pred_proba = model.predict(feature_vector, verbose=0)
                pred_class = np.argmax(pred_proba, axis=1)[0]
                confidence = np.max(pred_proba)
            else:
                pred_class = model.predict(feature_vector)[0]
                pred_proba = model.predict_proba(feature_vector)[0]
                confidence = np.max(pred_proba)
            
            return pred_class, confidence
        except Exception as e:
            print(f"      Error predicting with {model_name}: {str(e)[:100]}")
            return None, None
    
    def classify(self, df_temp, df_vib):
        """
        Classify bearing condition from CSV data using all available models
        """
        # Check if models are loaded
        if not self.models_loaded:
            print("   Warning: No classification models available")
            return {}
        
        feature_scaler = self.scalers.get('feature_scaler')
        label_encoder = self.scalers.get('label_encoder')
        
        if feature_scaler is None or label_encoder is None or self.feature_columns is None:
            print("   Warning: Preprocessing objects not loaded")
            return {}
        
        print("Preprocessing data for classification...")
        # Merge dataframes
        df_temp_clean = df_temp.drop(columns=['timestamp'], errors='ignore')
        df_vib_clean = df_vib.drop(columns=['timestamp'], errors='ignore')
        df_merged = pd.concat([df_temp_clean, df_vib_clean], axis=1)
        
        print(f"  Merged data shape: {df_merged.shape}")
        print(f"  Columns: {df_merged.columns.tolist()}")
        
        # Store results for all models
        all_model_results = {}
        
        # Get list of available models
        available_models = [k for k, v in self.models.items() if v is not None]
        
        if not available_models:
            print("  No models available for classification")
            return {}
        
        print(f"\nProcessing with {len(available_models)} available model(s): {', '.join(available_models)}")
        
        # Process with all available models
        for model_name in available_models:
            print(f"\n  Processing with {model_name}...")
            all_predictions = []
            all_confidences = []
            
            try:
                # Process each column
                for col_idx, col_name in enumerate(df_merged.columns):
                    signal_data = df_merged[col_name].values
                    
                    # Downsample from 25600 Hz to 1000 Hz
                    downsampled = self.downsample_signal(signal_data, original_fs=25600, target_fs=1000)
                    
                    # Segment (1000 samples = 1 second at 1000 Hz)
                    segments = self.segment_signal(downsampled, segment_length=1000, overlap=0.5)
                    
                    print(f"    Processing {col_name}: {len(segments)} segments")
                    
                    # Extract features and predict for each segment
                    for segment in segments:
                        features = self.extract_features_from_signal(segment, fs=1000)
                        
                        # Create feature vector in correct order matching training
                        feature_vector = []
                        for feat_name in self.feature_columns:
                            if feat_name in features:
                                feature_vector.append(features[feat_name])
                            else:
                                # Handle missing features
                                feature_vector.append(0)
                        
                        feature_vector = np.array(feature_vector).reshape(1, -1)
                        
                        # Verify feature count
                        if feature_vector.shape[1] != feature_scaler.n_features_in_:
                            print(f"       Feature mismatch: got {feature_vector.shape[1]}, expected {feature_scaler.n_features_in_}")
                            continue
                        
                        # Scale
                        feature_vector_scaled = feature_scaler.transform(feature_vector)
                        
                        # Predict
                        pred_class, confidence = self.predict_single_model(feature_vector_scaled, model_name)
                        
                        if pred_class is not None:
                            all_predictions.append(pred_class)
                            all_confidences.append(confidence)
                
                # Aggregate predictions (majority vote)
                if all_predictions:
                    unique, counts = np.unique(all_predictions, return_counts=True)
                    final_prediction = unique[np.argmax(counts)]
                    final_class_name = label_encoder.inverse_transform([final_prediction])[0]
                    avg_confidence = np.mean(all_confidences)
                    
                    # Parse fault information
                    if 'Normal' in final_class_name:
                        fault_type = 'Normal'
                        severity = 'None'
                    else:
                        parts = final_class_name.split('_')
                        fault_type = parts[0]
                        severity = '_'.join(parts[1:]) if len(parts) > 1 else 'Unknown'
                    
                    # Get maintenance recommendation
                    maintenance, fault_desc = self.get_maintenance_recommendation(fault_type, severity)
                    
                    # Store results for this model
                    all_model_results[model_name] = {
                        'predicted_class': final_class_name,
                        'fault_type': fault_type,
                        'fault_description': fault_desc,
                        'severity': severity,
                        'confidence': float(avg_confidence),
                        'maintenance_recommendation': maintenance,
                        'total_segments_analyzed': len(all_predictions)
                    }
                    
                    print(f"      {model_name}: {final_class_name} (Confidence: {avg_confidence:.2%})")
                else:
                    print(f"      {model_name}: No predictions generated")
                    
            except Exception as e:
                print(f"      {model_name} failed: {str(e)[:100]}")
                continue
        
        if all_model_results:
            print(f"\n  Classification completed with {len(all_model_results)} model(s)")
        else:
            print("\n  No classification results generated")
        
        return all_model_results


# Global classifier instance
_classifier = None

def get_classifier():
    """Get or create global classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = BearingClassifier()
    return _classifier

def classify_bearing(df_temp, df_vib):
    """
    Main function to classify bearing condition
    Returns empty dict if models not available
    """
    try:
        classifier = get_classifier()
        if not classifier.models_loaded:
            print("   Classification skipped: Models not loaded")
            return {}
        return classifier.classify(df_temp, df_vib)
    except Exception as e:
        print(f"  Classification error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}