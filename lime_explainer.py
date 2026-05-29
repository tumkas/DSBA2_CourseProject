import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime import lime_image
from cnn import HeartCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HeartCNN()
model.load_state_dict(torch.load('ultimate_model.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()

df = pd.read_csv('processed_data.csv')
abnormal_patients = df[df['label'] == -1]
sample_path = abnormal_patients.iloc[0]['npy_path']
mel_spec = np.load(sample_path)

mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
mel_resized = F.interpolate(mel_tensor, size=(224, 224), mode='bilinear', align_corners=False).squeeze().numpy()

image_for_lime = np.stack([mel_resized]*3, axis=-1)

def batch_predict(images):
    gray_images = images[:, :, :, 0]
    tensor_images = torch.tensor(gray_images, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        outputs = model(tensor_images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        return np.hstack((1 - probs, probs))

explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(
    image_for_lime,
    batch_predict,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)

ind = explanation.top_labels[0]
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.zeros_like(explanation.segments, dtype=float)

for k, v in dict_heatmap.items():
    heatmap[explanation.segments == k] = v

plt.figure(figsize=(10, 6))
plt.imshow(mel_resized, aspect='auto', cmap='viridis', origin='lower')
plt.imshow(heatmap, aspect='auto', cmap='RdBu_r', alpha=0.5, origin='lower')
plt.colorbar(label='LIME Weight (Red = Abnormal, Blue = Normal)')
plt.title("LIME Heatmap for Abnormal Heart Sound")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")
plt.tight_layout()
plt.show()