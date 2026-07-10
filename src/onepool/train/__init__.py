"""Training: LoRA fine-tuning structured as DiLoCo rounds from day one."""

TRAIN_INSTALL_HINT = (
    "training needs extra packages:\n"
    "  1. install PyTorch for your hardware:  onepool doctor\n"
    '  2. install the training stack:         pip install "onepool[train]"'
)


def require_training_stack() -> None:
    """Fail fast with instructions when the optional training deps are absent."""
    missing = []
    for module in ("torch", "transformers", "peft", "datasets"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise ImportError(f"missing: {', '.join(missing)}\n{TRAIN_INSTALL_HINT}")
