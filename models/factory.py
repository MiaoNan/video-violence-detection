"""Model construction, with explicit errors for unavailable architectures."""
import importlib
import torch


def VioNet_densenet_lean(config):
    # The original DenseNet implementation must be supplied separately.
    backend_name = getattr(config, "densenet_backend", "densenet")
    try:
        backend = importlib.import_module(backend_name)
    except ModuleNotFoundError as exc:
        if exc.name != backend_name:
            raise
        raise ModuleNotFoundError(
            f"Missing DenseNet backend {backend_name!r}. Restore the original "
            "module exposing densenet88 and get_fine_tuning_params, or set "
            "config.densenet_backend to its module name."
        ) from exc
    size = config.sample_size
    sample_size = size if isinstance(size, int) else size[0]
    model = backend.densenet88(
        num_classes=2, sample_size=sample_size,
        sample_duration=config.sample_duration,
    ).to(config.device)
    checkpoint_path = getattr(config, "checkpoint_path", "weights/weights.pth")
    state_dict = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(state_dict)
    params = backend.get_fine_tuning_params(model, config.ft_begin_idx)
    return model, params


def build_model(config):
    if config.model != "densenet_lean":
        raise ValueError(
            f"Model {config.model!r} is not implemented in this checkout; "
            "only the densenet_lean factory is present."
        )
    return VioNet_densenet_lean(config)
