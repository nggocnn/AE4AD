import argparse

from src.defenses.config_parser import AE4AD_Config
from src.defenses.autoencoder import AE4AD_Autoencoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configurations file path for AE4AD')

    parser.add_argument('--config_filepath', type=str, help='Required configurations file path')

    args = parser.parse_args()
    adversarial_config = AE4AD_Config(args.config_filepath)
    adversarial_generator = AE4AD_Autoencoder(adversarial_config)
    adversarial_generator.train(save_final_epoch=True)
