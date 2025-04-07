from pathlib import Path
from typing import Union, Type, List, Tuple
import pydoc

import numpy as np
import torch
from torch import autocast, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from batchgenerators.utilities.file_and_folder_operations import load_json
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

# Imported for BP-nnUNet
from nnunetv2.utilities.bp_nnunet import MetaTensor, load_blood_pressure, get_blood_pressure, get_embeddings, batchify_embed, to_batch_meta_tensor


DIM_BP = 5
DIM_TEXT_EMB = 512


class BP_nnUNet(nn.Module):
    """
    BP-nnUNet based on: "Kwon et al.,
    Blood Pressure Assisted Cerebral Microbleed Segmentation via Meta-matching
    <to be filled>"

    Part of the codes are referred from:
    nnU-Net based on: "Isensee et al.,
    nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation
    <https://www.nature.com/articles/s41592-020-01008-z>"
    UniSeg based on: "Ye et al.,
    UniSeg: A Prompt-Driven Universal Segmentation Model as Well as A Strong Representation Learner
    <https://link.springer.com/chapter/10.1007/978-3-031-43898-1_49>"
    """
    def __init__(self,
                 input_patch_size: List[int],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                    f"resolution stages. here: {n_stages}. " \
                                                    f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        # BP-nnUNet: expect two prompts: lobar and deep joint prompt
        # This step is necessary in advance to instantiate UnetDecoder.
        num_prompts = 2
        self.encoder.output_channels[-1] += num_prompts
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        # BP-nnUNet: Anatomically-aware Joint Prompt Fusion
        num_conv_stage = 3
        # get the shape of latent space from stage_plans
        num_pool_per_axis = np.array(strides, dtype=int)
        cum_strides = np.cumprod(num_pool_per_axis, axis=0)
        self.patch_size = np.array(input_patch_size, dtype=int) // cum_strides[-1]
        # get the flattened dimension of latent space
        patch_dim = np.prod(self.patch_size)
        # linear projection layers (from embeddings to latent space)
        self.text_to_vision = nn.Linear(DIM_TEXT_EMB, patch_dim)
        self.joint_to_vision = nn.Linear(DIM_TEXT_EMB + DIM_BP, patch_dim)

        # prompt fusion layer
        total_input_features = features_per_stage[-1]
        bottleneck_features = total_input_features // 4
        self.fusion_layer = StackedConvBlocks(num_conv_stage, conv_op, total_input_features,
                                              [bottleneck_features, bottleneck_features, num_prompts],
                                              [3, 3, 3], [1, 1, 1], False,
                                              norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin,
                                              nonlin_kwargs, nonlin_first=False)

    def forward(self, x):
        # BP-nnUNet: Prepare input data
        assert isinstance(x, MetaTensor), "Input is not MetaTensor"
        lobar = batchify_embed(x, "lobar_embed")
        deep = batchify_embed(x, "deep_embed")
        bp = x.metadata["bp"]
        joint = torch.cat([deep, bp], dim=1)

        skips = self.encoder(x)
        return_skips = self.encoder.return_skips
        x_feat = skips[-1] if return_skips else skips

        # BP-nnUNet: Anatomically-aware Joint Prompt Fusion
        ps = self.patch_size
        lobar_prompt = self.text_to_vision(lobar).reshape(-1, 1, ps[0], ps[1], ps[2])
        deep_prompt = self.joint_to_vision(joint).reshape(-1, 1, ps[0], ps[1], ps[2])
        joint_prompt = torch.cat([lobar_prompt, deep_prompt], dim=1)
        fused_output = self.fusion_layer(torch.cat([x_feat, joint_prompt], dim=1))

        x_feat = torch.cat([x_feat, fused_output], dim=1)

        if return_skips:
            skips[-1] = x_feat
        else:
            skips = x_feat

        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size) + \
            self.fusion_layer.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


class BP_Trainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.bp, self.embeds = None, None

    @staticmethod
    def _build_network_architecture(arch_kwargs: dict,
                                    arch_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                    input_channels: int,
                                    output_channels: int,
                                    patch_size: List[int],
                                    enable_deep_supervision: bool = True) -> nn.Module:
        architecture_kwargs = dict(**arch_kwargs)
        for ri in arch_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        if enable_deep_supervision is not None:
            architecture_kwargs['deep_supervision'] = enable_deep_supervision

        network = BP_nnUNet(
            input_patch_size=patch_size,
            input_channels=input_channels,
            num_classes=output_channels,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        # BP-nnUNet: input patch size is missing in this function.
        # Please use _build_network_architecture() instead, and pass the input patch size from configuration manager.
        raise NotImplementedError("This function has missing \"patch_size\" parameter. "
                                  "Please use _build_network_architecture() instead.")

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            # BP-nnUNet: Replace Residual nnU-Net to BP-nnUNet
            self.network = self._build_network_architecture(
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.configuration_manager.patch_size,
                self.enable_deep_supervision
            ).to(self.device)
            self.bp = load_blood_pressure()
            self.embeds = get_embeddings()
            
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)

            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        # BP-nnUNet: Extract blood pressure data
        bp = get_blood_pressure(self.bp, batch)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        bp = bp.to(self.device, non_blocking=True)
        # BP-nnUNet: Wrap BP into data tensor
        data = to_batch_meta_tensor(data, bp, self.embeds)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        # BP-nnUNet: Extract blood pressure data
        bp = get_blood_pressure(self.bp, batch)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        bp = bp.to(self.device, non_blocking=True)
        # BP-nnUNet: Wrap BP into data tensor
        data = to_batch_meta_tensor(data, bp, self.embeds)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
