"""Configuration-driven local causal-language-model loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelUnavailableError(RuntimeError):
    """Raised when local inference dependencies or hardware are unavailable."""


class GenerationSettings(BaseModel):
    id: str
    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(ge=0)
    do_sample: bool = False
    enable_thinking: bool = False


class RuntimeSettings(BaseModel):
    device: str = "auto"


class InferenceConfig(BaseModel):
    model: GenerationSettings
    runtime: RuntimeSettings


def load_inference_config(path: Path) -> InferenceConfig:
    """Load and validate model settings from YAML."""
    with path.open(encoding="utf-8") as config_file:
        return InferenceConfig.model_validate(yaml.safe_load(config_file))


def resolve_device(torch_module: Any, requested: str = "auto") -> str:
    """Resolve auto devices in CUDA, MPS, CPU priority order."""
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class LocalModel:
    tokenizer: Any
    model: Any
    device: str
    config: InferenceConfig

    def generate(self, prompt: str) -> str:
        """Generate only newly produced text, never hidden reasoning state."""
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.config.model.enable_thinking,
        ).to(self.device)
        generation_options: dict[str, Any] = {
            "max_new_tokens": self.config.model.max_new_tokens,
            "do_sample": self.config.model.do_sample,
        }
        if self.config.model.do_sample:
            generation_options.update(
                temperature=self.config.model.temperature,
                top_p=self.config.model.top_p,
                top_k=self.config.model.top_k,
            )
        output = self.model.generate(**inputs, **generation_options)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return str(self.tokenizer.decode(generated, skip_special_tokens=True))


def load_local_model(config: InferenceConfig) -> LocalModel:
    """Load the configured model on the best available local device."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        message = "ML support is not installed; run `uv sync --extra ml`"
        raise ModelUnavailableError(message) from exc

    device = resolve_device(torch, config.runtime.device)
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.model.id)
    model = AutoModelForCausalLM.from_pretrained(config.model.id, dtype=dtype)
    model.to(device)
    model.eval()
    return LocalModel(tokenizer=tokenizer, model=model, device=device, config=config)
