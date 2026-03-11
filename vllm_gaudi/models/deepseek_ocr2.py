from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_ocr2 import (
    DeepseekOCR2DummyInputsBuilder,
    DeepseekOCR2ForCausalLM,
    DeepseekOCR2MultiModalProcessor,
    DeepseekOCR2ProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekOCR2MultiModalProcessor,
    info=DeepseekOCR2ProcessingInfo,
    dummy_inputs=DeepseekOCR2DummyInputsBuilder,
)
class HpuDeepseekOCR2ForCausalLM(DeepseekOCR2ForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
