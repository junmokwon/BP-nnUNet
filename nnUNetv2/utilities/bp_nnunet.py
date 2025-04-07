"""
    BP-nnUNet based on: "Kwon et al.,
    Blood Pressure Assisted Cerebral Microbleed Segmentation via Meta-matching
    <to be filled>"

    Part of the codes are referred from:
    nnU-Net based on: "Isensee et al.,
    nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation
    <https://www.nature.com/articles/s41592-020-01008-z>"
"""
from pathlib import Path
import os
import numpy as np
import torch

from batchgenerators.utilities.file_and_folder_operations import load_json


# Simple tensor class to feed additional input to an existing framework.
class MetaTensor(torch.Tensor):
    def __new__(cls, data, metadata=None):
        instance = torch.as_tensor(data).as_subclass(cls)
        return instance

    def __init__(self, data, metadata=None):
        self.metadata = metadata or {}

    def __repr__(self):
        return f"MetaTensor({super().__repr__()}, metadata={self.metadata})"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        result = super().__torch_function__(func, types, args, kwargs)

        if isinstance(result, torch.Tensor):
            result = result.as_subclass(MetaTensor)
            result.metadata = getattr(self, "metadata", {}).copy()
        return result


# Read the blood pressure data from json file
def load_blood_pressure(bp_path=None):
    BP_DICT = {}
    if bp_path is None:
        env_name = "BP_nnUNet_BP_JSON"
        bp_path = os.environ.get(env_name)
        assert bp_path, f"Please specify \"{env_name}\" to read BP in json."
    meta = load_json(bp_path)
    for case, info in meta.items():
        assert "phenotypes" in info, f"Error in \"{bp_path}\": Case \"{case}\" does not have \"phenotypes\" array."
        phenotypes = np.array(info["phenotypes"], dtype=float)
        assert len(phenotypes) == 67, f"Incorrect phenotypes for case \"{case}\"; Desired length: 67, Real length: {len(phenotypes)}"
        bp = phenotypes[60:65].tolist()
        BP_DICT[case] = bp
    return BP_DICT


# Get the blood pressure data based on subject name
def get_blood_pressure(BP_DICT, batch, key_name='keys'):
    cases = batch[key_name]
    bp_list = []
    for case in cases:
        assert case in BP_DICT, f"Case \"{case}\" not found in BP json"
        bp = BP_DICT[case]
        bp_list.append(bp)
    bp_list = torch.tensor(bp_list, dtype=torch.float32)
    if torch.cuda.is_available():
        bp_list = bp_list.cuda()
    return bp_list


# Read the text embeddings from the embedding directory
def get_embeddings(embed_path=None):
    EMBED_DICT = {}
    if not EMBED_DICT:
        if embed_path is None:
            env_name = "BP_nnUNet_text_embed"
            embed_path = os.environ.get(env_name)
            assert embed_path, f"Please specify \"{env_name}\" to read text embeddings."
        mapper = {
            "cmb": "cerebral_microbleed",
            "deep": "deep_microbleed",
            "lobar": "lobar_microbleed"
        }
        for key_name, filename in mapper.items():
            embed = torch.load(os.path.join(embed_path, f"{filename}.pth")).reshape(1, 512)
            if torch.cuda.is_available():
                embed = embed.cuda()
            EMBED_DICT[key_name] = embed
    return EMBED_DICT


# Check if MetaTensor instantiation is necessary
def check_bp_nnunet():
    env_names = ("BP_nnUNet_BP_JSON", "BP_nnUNet_text_embed")
    for env_name in env_names:
        if env_name not in os.environ:
            return False
    return True


# Instantiate MetaTensor
def to_meta_tensor(_tensor, subject_name, bp_dict, embeds):
    bp = get_blood_pressure(bp_dict, {"keys": [subject_name]})
    metadata = {
        "bp": bp,
        "cmb_embed": embeds["cmb"],
        "deep_embed": embeds["deep"],
        "lobar_embed": embeds["lobar"]
    }
    return MetaTensor(_tensor, metadata)


def to_batch_meta_tensor(_tensor, bp, embeds):
    metadata = {
        "bp": bp,
        "cmb_embed": embeds["cmb"],
        "deep_embed": embeds["deep"],
        "lobar_embed": embeds["lobar"]
    }
    return MetaTensor(_tensor, metadata)


def batchify_embed(x, key):
    return x.metadata[key].repeat(x.size(0), 1)
