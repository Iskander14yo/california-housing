import logging
import time
from logging import getLogger
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from .utils.dvc import DVCManager
from .utils.sio import safe_skops_load

logger = getLogger()
logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="configs", config_name="default", version_base="1.2")
def infer(cfg: DictConfig):
    dvc_manager = DVCManager()

    logger.info("Downloading pipeline..")
    dvc_manager.pull(cfg.storage.paths.models.folder)
    preprocessor = safe_skops_load(
        Path(cfg.storage.paths.models.folder) / "CH_preprocessor.skops",
        add_types=["california_housing"],
    )
    model = safe_skops_load(
        Path(cfg.storage.paths.models.folder) / "CH_model.skops",
        add_types=["california_housing"],
    )

    logger.info("Downloading inference data..")
    infer_path = Path(cfg.storage.paths.data.folder) / cfg.storage.paths.data.test_path
    pred_path = (
        Path(cfg.storage.paths.data.folder) / ("CH_predicts_" + str(int(time.time())))
    ).with_suffix(".csv")
    dvc_manager.pull(infer_path)
    test_df = pd.read_csv(infer_path, index_col=0)

    logger.info("Making predictions..")
    test_df_proc = preprocessor.process_infer(test_df)
    y_pred = model.predict(test_df_proc)
    df_pred = pd.DataFrame(y_pred)
    df_pred.to_csv(pred_path)

    logger.info(f"Inference successfully made! Predictions are saved to {pred_path}.")


if __name__ == "__main__":
    infer()
