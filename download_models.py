import os

from huggingface_hub import snapshot_download
from utils import AVAILABLE_MODELS, HF_MODEL_MAPPING, logger


def download_models() -> None:
    os.makedirs("./models", exist_ok=True)

    for model_name, local_path in AVAILABLE_MODELS.items():
        logger.info(f"Downloading model: {model_name}")

        repo_id = HF_MODEL_MAPPING[model_name]

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                cache_dir=None,
            )
            logger.info(f"Model {model_name} downloaded to {local_path}")

        except Exception as e:
            logger.error(f"Error downloading {model_name}: {e}")


if __name__ == "__main__":
    download_models()
