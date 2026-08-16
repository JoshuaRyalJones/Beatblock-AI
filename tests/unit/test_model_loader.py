from pathlib import Path
from types import SimpleNamespace

from beatblock.model.loader import load_inference_config, resolve_device


def _torch(cuda: bool = False, mps: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def test_loads_model_id_from_configuration() -> None:
    config = load_inference_config(Path("configs/model.yaml"))

    assert config.model.id == "Qwen/Qwen3-1.7B"
    assert config.model.do_sample is False
    assert config.model.enable_thinking is False


def test_device_resolution_priority() -> None:
    assert resolve_device(_torch(cuda=True, mps=True)) == "cuda"
    assert resolve_device(_torch(mps=True)) == "mps"
    assert resolve_device(_torch()) == "cpu"
    assert resolve_device(_torch(), "cpu") == "cpu"
