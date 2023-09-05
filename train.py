import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# load environmental variables in .env file
load_dotenv()

import hydra
from hydra.utils import instantiate

from src.groovis.schemas import Cfg

# from src.groovis.models.architectures import Architecture


# import config automatically by hydra.main
@hydra.main(config_path="configs", config_name="default", version_base="1.3.2")
def main(config: Cfg):
    architecture = instantiate(config.architecture)
    # architecture = Architecture(
    #     patch_size=config.architecture.patch_size,
    #     channels=config.architecture.channels,
    #     embed_dim=config.architecture.embed_dim,
    # )
    # print(config)
    print(architecture)
    # train(config=config)


if __name__ == "__main__":
    main()
