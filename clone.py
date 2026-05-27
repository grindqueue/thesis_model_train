from huggingface_hub import hf_hub_download
import tensorflow as tf, json

REPO = "damilareisaac/parental-control-efficientnet-b0"
model = tf.keras.models.load_model(hf_hub_download(REPO, "parental_control_b0.keras"))