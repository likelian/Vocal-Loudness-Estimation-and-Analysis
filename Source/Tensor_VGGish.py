import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load the model.
model = hub.load('https://tfhub.dev/google/vggish/1')

# Input: 3 seconds of silence as mono 16 kHz waveform samples.
waveform = np.zeros(3 * 16000, dtype=np.float32)

# Run the model, check the output.
embeddings = model(waveform)
embeddings_arr = embeddings.numpy()
print(embeddings_arr)
#embeddings.shape.assert_is_compatible_with([None, 128])
