"""
classify.py - Bearing Fault Classification Module
Using Ensemble Model (Random Forest + XGBoost + LightGBM + SVM + Gradient Boosting)
Matches exact training pipeline: 2560 Hz, 2560 samples, 50% overlap
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew, entropy
from scipy.fft import fft, fftfreq
import joblib
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

class BearingClassifier:
    """Ensemble-based classifier for bearing fault detection"""

    def __init__(self):
        """Initialize classifier with ensemble model paths"""
        self.ensemble_model = None
        self.feature_scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.models_loaded = False
        self.load_ensemble_model()

    def load_ensemble_model(self):
        """Load ensemble model and preprocessing objects"""
        try:
            # Define paths to ensemble model files
            ENSEMBLE_DIR = 'ensemble'
            MODEL_PATH = r"ensemble/ensemble_bearing_fault_classifier.pkl"
            SCALER_PATH = r"ensemble/ensemble_feature_scaler.pkl"
            ENCODER_PATH = r"ensemble/ensemble_label_encoder.pkl"
            FEATURES_PATH = r"ensemble/ensemble_feature_columns.pkl"

            print("\n" + "="*60)
            print("LOADING ENSEMBLE CLASSIFICATION MODEL")
            print("="*60)

            # Load ensemble model
            print("Loading ensemble model...")
            self.ensemble_model = joblib.load(MODEL_PATH)
            print("  ✓ Ensemble model loaded (Voting Classifier)")

            # Load feature scaler
            print("Loading feature scaler...")
            self.feature_scaler = joblib.load(SCALER_PATH)
            print(f"  ✓ Feature scaler loaded (expects {self.feature_scaler.n_features_in_} features)")

            # Load label encoder
            print("Loading label encoder...")
            self.label_encoder = joblib.load(ENCODER_PATH)
            print(f"  ✓ Label encoder loaded ({len(self.label_encoder.classes_)} classes)")
            print(f"    Classes: {', '.join(self.label_encoder.classes_[:3])}... (+{len(self.label_encoder.classes_)-3} more)")

            # Load feature columns
            print("Loading feature columns...")
            self.feature_columns = joblib.load(FEATURES_PATH)
            print(f"  ✓ Feature columns loaded ({len(self.feature_columns)} features)")

            self.models_loaded = True
            print("\n✓ Ensemble model loaded successfully!")
            print("="*60 + "\n")
            return True

        except FileNotFoundError as e:
            print(f"\n✗ Error: Model files not found")
            print(f"  Missing file: {e.filename}")
            print(f"  Please ensure all ensemble model files are in the '{ENSEMBLE_DIR}' directory")
            print("="*60 + "\n")
            self.models_loaded = False
            return False

        except Exception as e:
            print(f"\n✗ Critical error loading ensemble model: {e}")
            print("="*60 + "\n")
            self.models_loaded = False
            return False

    def downsample_signal(self, signal_data, original_fs=25600, target_fs=2560):
        """
        Downsample signal from 25600 Hz to 2560 Hz (matches training)
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

    def segment_signal(self, signal_data, segment_length=2560, overlap=0.5):
        """
        Segment signal into overlapping windows
        segment_length: 2560 samples (1 second at 2560 Hz) - matches training
        overlap: 0.5 (50% overlap) - matches training
        """
        step = int(segment_length * (1 - overlap))
        segments = []

        for i in range(0, len(signal_data) - segment_length + 1, step):
            segments.append(signal_data[i:i + segment_length])

        return np.array(segments)

    def extract_time_domain_features(self, signal_data):
        """
        Extract time-domain statistical features
        Exactly matches training feature extraction
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

    def extract_frequency_domain_features(self, signal_data, fs=2560):
        """
        Extract frequency-domain features using FFT
        fs=2560 Hz - matches training sampling rate
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

    def extract_features_from_signal(self, signal_data, fs=2560):
        """
        Extract all features from a signal
        Returns dict with time_ and freq_ prefixed features
        """
        features = {}

        # Time domain features
        time_features = self.extract_time_domain_features(signal_data)
        features.update({f'time_{k}': v for k, v in time_features.items()})

        # Frequency domain features (fs=2560 Hz)
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

    def classify(self, df_temp, df_vib):
        """
        Classify bearing condition using ensemble model
        Follows exact training pipeline order and parameters
        """
        # Check if model is loaded
        if not self.models_loaded:
            print("  ✗ Warning: Ensemble model not available")
            return {}

        print("Preprocessing data for ensemble classification...")

        # Clean dataframes (remove timestamp columns)
        df_temp_clean = df_temp.drop(columns=['timestamp'], errors='ignore')
        df_vib_clean = df_vib.drop(columns=['timestamp'], errors='ignore')

        print(f"  Temperature/Current data shape: {df_temp_clean.shape}")
        print(f"  Vibration data shape: {df_vib_clean.shape}")

        # Store all predictions and confidences
        all_predictions = []
        all_confidences = []
        all_probabilities = []

        # PROCESS CURRENT/TEMP DATA FIRST (following training order)
        print("\n  Processing current/temperature data...")
        for col_idx, col_name in enumerate(df_temp_clean.columns):
            signal_data = df_temp_clean[col_name].values

            # Downsample from 25600 Hz to 2560 Hz
            downsampled = self.downsample_signal(signal_data, original_fs=25600, target_fs=2560)

            # Segment (2560 samples = 1 second at 2560 Hz, 50% overlap)
            segments = self.segment_signal(downsampled, segment_length=2560, overlap=0.5)

            print(f"    {col_name}: {len(segments)} segments created")

            # Extract features and predict for each segment
            for seg_idx, segment in enumerate(segments):
                features = self.extract_features_from_signal(segment, fs=2560)

                # Add metadata (matching training format)
                features['sensor'] = f'current_temp_{col_name}'
                features['segment_id'] = seg_idx
                features['data_type'] = 'current_temp'

                # Create feature vector in correct order
                feature_vector = []
                for feat_name in self.feature_columns:
                    # Skip metadata columns
                    if feat_name in ['sensor', 'segment_id', 'file_name', 'data_type']:
                        continue

                    if feat_name in features:
                        feature_vector.append(features[feat_name])
                    else:
                        # Handle missing features (shouldn't happen if extraction matches training)
                        feature_vector.append(0)

                feature_vector = np.array(feature_vector).reshape(1, -1)

                # Replace NaN and Inf with 0 (matching training preprocessing)
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

                # Scale features
                try:
                    feature_vector_scaled = self.feature_scaler.transform(feature_vector)

                    # Predict using ensemble model
                    pred_class = self.ensemble_model.predict(feature_vector_scaled)[0]
                    pred_proba = self.ensemble_model.predict_proba(feature_vector_scaled)[0]
                    confidence = np.max(pred_proba)

                    all_predictions.append(pred_class)
                    all_confidences.append(confidence)
                    all_probabilities.append(pred_proba)

                except Exception as e:
                    print(f"      ✗ Error predicting segment {seg_idx}: {str(e)[:50]}")
                    continue

        # THEN PROCESS VIBRATION DATA (following training order)
        print("\n  Processing vibration data...")
        for col_idx, col_name in enumerate(df_vib_clean.columns):
            signal_data = df_vib_clean[col_name].values

            # Downsample from 25600 Hz to 2560 Hz
            downsampled = self.downsample_signal(signal_data, original_fs=25600, target_fs=2560)

            # Segment (2560 samples = 1 second at 2560 Hz, 50% overlap)
            segments = self.segment_signal(downsampled, segment_length=2560, overlap=0.5)

            print(f"    {col_name}: {len(segments)} segments created")

            # Extract features and predict for each segment
            for seg_idx, segment in enumerate(segments):
                features = self.extract_features_from_signal(segment, fs=2560)

                # Add metadata (matching training format)
                features['sensor'] = f'vibration_{col_name}'
                features['segment_id'] = seg_idx
                features['data_type'] = 'vibration'

                # Create feature vector in correct order
                feature_vector = []
                for feat_name in self.feature_columns:
                    # Skip metadata columns
                    if feat_name in ['sensor', 'segment_id', 'file_name', 'data_type']:
                        continue

                    if feat_name in features:
                        feature_vector.append(features[feat_name])
                    else:
                        # Handle missing features
                        feature_vector.append(0)

                feature_vector = np.array(feature_vector).reshape(1, -1)

                # Replace NaN and Inf with 0
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

                # Scale features
                try:
                    feature_vector_scaled = self.feature_scaler.transform(feature_vector)

                    # Predict using ensemble model
                    pred_class = self.ensemble_model.predict(feature_vector_scaled)[0]
                    pred_proba = self.ensemble_model.predict_proba(feature_vector_scaled)[0]
                    confidence = np.max(pred_proba)

                    all_predictions.append(pred_class)
                    all_confidences.append(confidence)
                    all_probabilities.append(pred_proba)

                except Exception as e:
                    print(f"      ✗ Error predicting segment {seg_idx}: {str(e)[:50]}")
                    continue

        # Aggregate predictions (majority vote)
        if not all_predictions:
            print("\n  ✗ No predictions generated")
            return {}

        print(f"\n  Total segments analyzed: {len(all_predictions)}")

        # Use majority voting
        unique, counts = np.unique(all_predictions, return_counts=True)
        final_prediction = unique[np.argmax(counts)]
        final_class_name = self.label_encoder.inverse_transform([final_prediction])[0]

        # Calculate average confidence
        avg_confidence = np.mean(all_confidences)

        # Calculate class distribution
        class_distribution = {}
        for class_idx, count in zip(unique, counts):
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            percentage = (count / len(all_predictions)) * 100
            class_distribution[class_name] = {
                'count': int(count),
                'percentage': float(percentage)
            }

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

        # Prepare result
        result = {
            'ensemble': {
                'predicted_class': final_class_name,
                'fault_type': fault_type,
                'fault_description': fault_desc,
                'severity': severity,
                'confidence': float(avg_confidence),
                'maintenance_recommendation': maintenance,
                'total_segments_analyzed': len(all_predictions),
                'class_distribution': class_distribution,
                'model_type': 'Ensemble (RF + XGB + LGB + SVM + GB)'
            }
        }

        print(f"\n  ✓ Classification complete!")
        print(f"    Predicted: {final_class_name}")
        print(f"    Confidence: {avg_confidence:.2%}")
        print(f"    Fault Type: {fault_desc}")

        return result


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
    Main function to classify bearing condition using ensemble model
    Returns empty dict if model not available
    """
    try:
        classifier = get_classifier()
        if not classifier.models_loaded:
            print("  ✗ Classification skipped: Ensemble model not loaded")
            return {}
        return classifier.classify(df_temp, df_vib)
    except Exception as e:
        print(f"  ✗ Classification error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}