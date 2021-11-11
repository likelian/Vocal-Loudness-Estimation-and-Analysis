import torch
import numpy as np

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

# Download an example audio file
import urllib
url, filename = ("http://soundbible.com/grab.php?id=1698&type=wav", "bus_chatter.wav")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)


waveform = np.ones(3 * 16000, dtype=np.float32)

#a = model.forward(filename)

print(filename)

a = model.forward(waveform)
print(a)
