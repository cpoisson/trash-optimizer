"""Base configuration for training."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BaseConfig:
    """Base configuration class for training."""

    # Dataset configuration
    DATASET_ROOT_DIR = os.getenv('DATASET_ROOT_DIR')
    RESULTS_ROOT_DIR = os.getenv('RESULTS_ROOT_DIR')

    # Data loading
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    TEST_SPLIT = 0.25
    RANDOM_SEED = 42

    # Training
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10

    # Model
    FREEZE_LAYERS = -3  # Freeze all except last 3 blocks

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if cls.DATASET_ROOT_DIR is None:
            raise ValueError("DATASET_ROOT_DIR environment variable is not set")
        if cls.RESULTS_ROOT_DIR is None:
            raise ValueError("RESULTS_ROOT_DIR environment variable is not set")

        # Ensure directories exist
        Path(cls.DATASET_ROOT_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.RESULTS_ROOT_DIR).mkdir(parents=True, exist_ok=True)
