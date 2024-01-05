import fire
from omegaconf import OmegaConf

from california_housing.infer import infer
from california_housing.train import train


class Pipeline:
    def __init__(self, cfg_path):
        # Load configuration using Hydra
        self.cfg = OmegaConf.load(cfg_path)  # Adjust path and filename as necessary

    def train(self):
        # Pass the configuration to the train function
        train(self.cfg)

    def inference(self):
        # Pass the configuration to the infer function
        infer(self.cfg)


if __name__ == "__main__":
    fire.Fire(Pipeline)
