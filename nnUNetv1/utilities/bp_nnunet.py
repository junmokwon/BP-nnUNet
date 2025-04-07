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


BP_DICT = {}
# Read the blood pressure data from json file
def get_blood_pressure(batch, key_name='keys', bp_path=None):
    if not BP_DICT:
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


EMBED_DICT = {}
# Read the text embeddings from the embedding directory
def get_embeddings(embed_path=None):
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
def to_meta_tensor(_tensor, subject_name):
    bp = get_blood_pressure({"keys": [subject_name]})
    embeds = get_embeddings()
    metadata = {
        "bp": bp,
        "cmb_embed": embeds["cmb"],
        "deep_embed": embeds["deep"],
        "lobar_embed": embeds["lobar"]
    }
    return MetaTensor(_tensor, metadata)


def to_batch_meta_tensor(_tensor, bp):
    embeds = get_embeddings()
    metadata = {
        "bp": bp,
        "cmb_embed": embeds["cmb"],
        "deep_embed": embeds["deep"],
        "lobar_embed": embeds["lobar"]
    }
    return MetaTensor(_tensor, metadata)


def nnunetv1_get_subject_name(file):
    _output_file = Path(file)
    name = _output_file.name[:-len(''.join(_output_file.suffixes))]
    return name


# Minor fix of "from batchgenerators.augmentations.utils import pad_nd_image"
# https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/utils.py
# In case of padding an image, we wrap a numpy array into the MetaTensor
# to ensure to keep their metadata.
def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    if kwargs is None:
        kwargs = {'constant_values': 0}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        # Keep metadata whenever possible.
        if hasattr(image, 'metadata'):
            metadata = image.metadata
            res = np.pad(image, pad_list, mode, **kwargs)
            res = MetaTensor(res, metadata)
        else:
            res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer


# Minor fix of "from nnunet.utilities.to_torch import maybe_to_torch"
# https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/nnunet/utilities/to_torch.py
# Keep the metadata when converting a numpy array into the MetaTensor.
def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        # Keep metadata whenever possible.
        if hasattr(d, 'metadata'):
            metadata = d.metadata
            d = MetaTensor(torch.from_numpy(d).float(), metadata=metadata)
        else:
            d = torch.from_numpy(d).float()
    return d


def batchify_embed(x, key):
    return x.metadata[key].repeat(x.size(0), 1)
