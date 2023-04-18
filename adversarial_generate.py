import argparse

from ae4ad.adversary.config_parser import AdversarialConfig
from ae4ad.adversary.generator import AdversarialGenerator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configurations file path for adversarial generating')

    parser.add_argument('--config_filepath', type=str, help='Required configurations file path')

    args = parser.parse_args()
    adversarial_config = AdversarialConfig(args.config_filepath)
    adversarial_generator = AdversarialGenerator(adversarial_config)
    adversarial_generator.run()

