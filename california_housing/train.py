import datetime
import logging
import os.path
from logging import getLogger
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from sklearn.model_selection import learning_curve, train_test_split
from skops import io as sio

from .models.model import Model
from .models.preprocessing import Preprocessor
from .utils.dvc import DVCManager
from .utils.git import get_head_sha

logger = getLogger()
logging.basicConfig(level=logging.INFO)

RUN_TIME = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
RUN_COMMIT = get_head_sha()


@hydra.main(config_path="configs", config_name="default", version_base="1.2")
def train(cfg: DictConfig):
    dvc_manager = DVCManager()

    logger.info("Downloading train data..")
    train_data_path = (
        Path(cfg.storage.paths.data.folder) / cfg.storage.paths.data.train_path
    )
    dvc_manager.pull(train_data_path)
    logger.info("Downloaded successfully!")

    logger.info("Running experiment..")
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment("California Housing")
    with mlflow.start_run(
        run_name=RUN_TIME + " || " + RUN_COMMIT,
    ):
        mlflow.log_params(cfg.pipeline)
        mlflow.log_param("git commit", RUN_COMMIT)

        # Запускаем предобработку
        logger.info("Start preprocessing..")
        train_df = pd.read_csv(train_data_path, index_col=0)
        preprocessor = Preprocessor(cfg.pipeline)
        X, y = preprocessor.process_train(train_df)
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.1,
            random_state=42,
            shuffle=True,
        )

        # Запускаем обучение
        logger.info("Start training..")
        model = Model(**cfg.pipeline.model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        logger.info("Ended training successfully!")

        logger.info("Plotting metrics..")
        # 1. Важность фичей
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame(
            {
                "Feature": preprocessor.infer_pipeline.get_feature_names_out(),
                "Importance": feature_importance,
            }
        )
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="Importance",
            y="Feature",
            hue="Feature",
            data=importance_df,
            palette="viridis",
        )
        plt.title("Feature Importance")
        plt.xlabel("Важность")
        plt.ylabel("Признак")
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        os.remove("feature_importance.png")
        logger.info("Plotted features importance")

        # 2. График сравнения реальных и предсказанных значений
        plt.figure(figsize=(10, 6))
        plt.plot(y_val, label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.title("Actual vs Predicted")
        plt.legend()
        plt.savefig("actual_vs_predicted.png")
        mlflow.log_artifact("actual_vs_predicted.png")
        os.remove("actual_vs_predicted.png")
        logger.info("Plotted comparison of true and predicted values")

        # 3. График обучения
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model,
            X=X,
            y=y,
            cv=5,
            train_sizes=np.linspace(0.1, 1.0, 5),
            verbose=0,
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.figure(figsize=(10, 6))
        plt.plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="Cross-validation score",
        )
        plt.title("Learning curve")
        plt.legend()
        plt.savefig("learning_curve.png")
        mlflow.log_artifact("learning_curve.png")
        os.remove("learning_curve.png")
        logger.info("Plotted learning curve of training process")

    # Сохранение модели и препроцессинга
    logger.info("Ended training model. Saving results..")
    os.makedirs(cfg.storage.paths.models.folder, exist_ok=True)
    preproc_path = Path(cfg.storage.paths.models.folder) / "CH_preprocessor.skops"
    model_path = Path(cfg.storage.paths.models.folder) / "CH_model.skops"
    sio.dump(preprocessor, preproc_path)
    sio.dump(model, model_path)
    dvc_manager.add(cfg.storage.paths.models.folder)
    logger.info(
        f"Experiment successfully run! Models are saved to {cfg.storage.paths.models.folder}."
    )


if __name__ == "__main__":
    train()
