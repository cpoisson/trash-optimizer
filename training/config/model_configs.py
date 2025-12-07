"""Model-specific configurations."""


MODEL_CONFIGS = {
    'efficientnet_b0': {
        'learning_rate': 0.0001,
        'batch_size': 32,
        'freeze_layers': -3,
        'description': 'EfficientNet-B0 (lightweight baseline)'
    },
    'efficientnet_b2': {
        'learning_rate': 0.0001,
        'batch_size': 32,
        'freeze_layers': -3,
        'description': 'EfficientNet-B2 (more capacity)'
    },
    'efficientnet_v2_s': {
        'learning_rate': 0.0001,
        'batch_size': 32,
        'freeze_layers': -3,
        'description': 'EfficientNetV2-S (improved efficiency)'
    },
    'efficientnet_v2_m': {
        'learning_rate': 0.0001,
        'batch_size': 24,  # Larger model, smaller batch
        'freeze_layers': -3,
        'description': 'EfficientNetV2-M (larger capacity)'
    },
    'convnext_tiny': {
        'learning_rate': 0.0001,
        'batch_size': 32,
        'freeze_layers': -2,
        'description': 'ConvNeXt-Tiny (modern architecture)'
    },
    'resnet50': {
        'learning_rate': 0.0001,
        'batch_size': 32,
        'freeze_layers': -2,
        'description': 'ResNet50 (classic architecture)'
    }
}


def get_model_config(model_name, base_config):
    """Get merged configuration for a specific model.

    Args:
        model_name: Name of the model.
        base_config: Base configuration class.

    Returns:
        dict: Merged configuration.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {', '.join(MODEL_CONFIGS.keys())}"
        )

    model_cfg = MODEL_CONFIGS[model_name].copy()

    # Merge with base config
    config = {
        'model_name': model_name,
        'dataset_root_dir': base_config.DATASET_ROOT_DIR,
        'results_root_dir': base_config.RESULTS_ROOT_DIR,
        'batch_size': model_cfg.get('batch_size', base_config.BATCH_SIZE),
        'num_workers': base_config.NUM_WORKERS,
        'test_split': base_config.TEST_SPLIT,
        'random_seed': base_config.RANDOM_SEED,
        'num_epochs': base_config.NUM_EPOCHS,
        'learning_rate': model_cfg.get('learning_rate', base_config.LEARNING_RATE),
        'weight_decay': base_config.WEIGHT_DECAY,
        'early_stopping_patience': base_config.EARLY_STOPPING_PATIENCE,
        'freeze_layers': model_cfg.get('freeze_layers', base_config.FREEZE_LAYERS),
        'description': model_cfg.get('description', '')
    }

    return config
