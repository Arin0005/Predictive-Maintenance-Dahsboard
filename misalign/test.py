from tensorflow.keras.models import load_model

# Load your model
model = load_model("misalign_predictive_maintenance_model.h5",compile=False)

# Check the input shape
print("Input shape:", model.input_shape)

# (Optional) check model summary
model.summary()
