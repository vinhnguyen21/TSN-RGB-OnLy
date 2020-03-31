import argparse
import json

from src.trainer import TSN_training

def main(args):
    params = json.load(open(args.config, 'r'))
    print(params)
    trainer = TSN_training(**params)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'UCF101-parameters-training')
    parser.add_argument('--config', default= './train_config/config.json', type = str, help = 'config file')
    args = parser.parse_args()
    main(args)