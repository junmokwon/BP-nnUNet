"""
    BP-nnUNet based on: "Kwon et al.,
    Blood Pressure Assisted Cerebral Microbleed Segmentation via Meta-matching
    <to be filled>"

    Part of the codes are referred from:
    CLIP-Driven Universal Model based on: "Liu et al.,
    CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection
    <https://ieeexplore.ieee.org/document/10376801>"
"""
from pathlib import Path
import open_clip
import torch


MODEL_TAG = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
OUTPUT_DIR = '/path/to/biomedclip/embeddings/'

model, _ = open_clip.create_model_from_pretrained(MODEL_TAG)
tokenizer = open_clip.get_tokenizer(MODEL_TAG)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

INFO = {
    'cerebral_microbleed': ['Cerebral Microbleed'],
    'deep_microbleed': ['Deep Microbleed'],
    'lobar_microbleed': ['Lobar Microbleed']
}

OUTPUT_DIR = Path(OUTPUT_DIR)
for filename, organ_name in INFO.items():
    with torch.no_grad():
        text_inputs = torch.cat([tokenizer(f'A magnetic resonance imaging of a {item}') for item in organ_name]).to(device)
        text_embed = model.encode_text(text_inputs, normalize=True).reshape(1, 512)
        torch.save(text_embed, str(OUTPUT_DIR / f'{filename}.pth'))
