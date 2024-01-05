import os
import time
from logging import getLogger
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from .utils.dvc import DVCManager
from .utils.sio import safe_skops_load

logger = getLogger()
dvc_manager = DVCManager()


@hydra.main(config_path="configs", config_name="default", version_base="1.2")
def infer(cfg: DictConfig):
    logger.info("Downloading pipeline..")
    dvc_manager.pull(cfg.storage.paths.models.folder)
    preprocessor = safe_skops_load(
        os.path.join(cfg.storage.paths.models.folder, "CH_preprocessor.skops"),
        add_types=["california_housing"],
    )
    model = safe_skops_load(
        os.path.join(cfg.storage.paths.models.folder, "CH_model.skops"),
        add_types=["california_housing"],
    )
    preprocessor.is_train = False

    logger.info("Downloading inference data..")
    infer_path = Path(cfg.storage.paths.data.folder) / cfg.storage.paths.data.test_path
    pred_path = (
        Path(cfg.storage.paths.data.folder) / ("CH_predicts_" + str(int(time.time())))
    ).with_suffix(".csv")
    dvc_manager.pull(infer_path)
    test_df = pd.read_csv(infer_path, index_col=0)

    logger.info("Making predictions..")
    test_df_proc = preprocessor.transform(test_df)
    y_pred = model.predict(test_df_proc)
    df_pred = pd.DataFrame(y_pred)
    df_pred.to_csv(pred_path)

    logger.info("Inference successfully made! Predictions are saved.")


if __name__ == "__main__":
    infer()
