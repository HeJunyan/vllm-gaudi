# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from types import MethodType

import torch
import torch.nn as nn

from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict
from vllm.model_executor.models.deepseek_ocr2 import (
    DeepseekOCR2ForCausalLM,
    DeepseekOCR2MultiModalProcessor,
    DeepseekOCR2ProcessingInfo,
    DeepseekOCR2DummyInputsBuilder,
)
from vllm.model_executor.models.deepencoder2 import (
    CustomQwen2Decoder,
)

import habana_frameworks.torch.core as htcore

class HpuDeepseekOCR2Visual(nn.Module):
    def __init__(
        self,
        sam_model,
        qwen2_model,
    ):
        super().__init__()
        self.sam_model = sam_model
        self.qwen2_model = qwen2_model
        self.patch_size = 16

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        htcore.mark_step()
        features_1 = self.sam_model(image_tensor)
        htcore.mark_step()
        features_2 = self.qwen2_model(features_1)
        return features_2


class HpuDeepseekOCR2DummyInputsBuilder(DeepseekOCR2DummyInputsBuilder):
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        max_image_size = self.info.get_image_size_with_most_features()

        # All possible deepseek ocr input is always
        #    pixel_values: torch.Size([1, 3, 1024, 1024])
        #    images_crop: torch.Size([sub_image_num, 3, 640, 640])
        #    images_spatial_crop: torch.Size([1, 2])
        # Where sub_image_num is in [0, 2, 3, 4, 5, 6], decided by the
        # resolution and the aspect ratio.
        # The follow code can imitate all the possible sub_image_num.
        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                num_images=num_images,
                width=max_image_size.width,
                height=max_image_size.height,
                overrides=image_overrides,
            )
        }


#Change the forward of CustomQwen2Decoder, we need to upload
#token_type_ids to HPU.
def custom_qwen2_decoder_forward(
    self,
    inputs_embeds: torch.Tensor,
    token_type_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    **kwargs,
):
    token_type_ids = token_type_ids.to(inputs_embeds.device)
    return self.model(
        inputs_embeds=inputs_embeds,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        **kwargs,
    )


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekOCR2MultiModalProcessor,
    info=DeepseekOCR2ProcessingInfo,
    dummy_inputs=HpuDeepseekOCR2DummyInputsBuilder,
)
class HpuDeepseekOCR2ForCausalLM(DeepseekOCR2ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.qwen2_model.model.forward = MethodType(custom_qwen2_decoder_forward, self.qwen2_model.model)
        self.visual = HpuDeepseekOCR2Visual(self.sam_model, self.qwen2_model)

    def _encode_global_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        global_features = self.visual(image_tensor)
        features = self.projector(global_features)

        _, hw, dim = features.shape

        return features.view(-1, dim)

    def _encode_local_features(self, patches: torch.Tensor) -> torch.Tensor | None:
        if torch.sum(patches).item() == 0:
            return None

        local_features = self.visual(patches)
        features = self.projector(local_features)

        _, _, dim = features.shape

        return features.view(-1, dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None

        # The deepseek model only accepts (token, embed) tensor type,
        # We need to flatten (batch, token, embed) into (batch*token, embed)
        # here. The ForwardContext hold the current batch info and
        # HPUAttentionImpl will view it as (batch, token, embed) again.
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
                inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-1))

        hidden_states = self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states
