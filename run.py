import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()  # load environmental variables in .env file

import hydra

from src.groovis.configs import Config, register_configs


# import config automatically by hydra.main
@hydra.main(config_name="default", version_base="1.3.2")
def main(config: Config):
    print(config)
    # train(config=config)


if __name__ == "__main__":
    register_configs()
    main()
