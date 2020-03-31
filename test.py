import argparse
import json
import torch
import os
import torchvision
from src.dataset_ucf import UCF_dataset
from src.trainer import TSN_training
import torch.utils.data as data
from tqdm import tqdm
from src.models import TSN
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

def main(args):
    params = json.load(open(args.config, 'r'))
    trainer = TSN_training(**params)

    #get testing dataset
    action = {}
    file1 = open(params['class_label'], "r").readlines()
    label_pre = [x.split('\n')[0] for x in file1]
    label_pre = [x.split(' ') for x in label_pre]
    for label in label_pre:
        action[label[1]] = int(label[0])

    # get filename, label for test_set
    folder_name = os.listdir(params['data_path'])
    test_sets = ["g01", "g02", "g03", "g04", "g05", "g06", "g07"]
    X_test = []
    y_test = []

    for f in folder_name:
        if f[-7:-4] in test_sets:
            loc1 = f.find('v_')
            loc2 = f.find('_g')
            temp_act = f[(loc1+2) : loc2]
            y_test.append(action[temp_act])
            X_test.append(f)        

    # checking dataset
    if params['dataset'] == 'ucf101':
        num_class = 101
    elif params['dataset'] == 'hmdb51':
        num_class = 51
    elif params['dataset'] == 'kinetics':
        num_class = 400
    else:
        raise ValueError('Unknown dataset ')
    
    #loading model
    if not os.path.exists(params['model_path']):
        raise ValueError('Wrong weight path')
    model, transform, optimizer, criterion, device = trainer.get_training_object(num_class)
    state_dict = torch.load(params['model_path'], map_location = device)

    state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)

    test_set = UCF_dataset(params['data_path'], X_test, y_test, num_segment= 3, transform = transform['valid'])
    test_loader = data.DataLoader(test_set, batch_size= params['batch_size'], shuffle=False, num_workers = 4, pin_memory= False)

    score = trainer._test(test_loader, model, device= device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'UCF101-parameters-training')
    parser.add_argument('--config', default= './train_config/config.json', type = str, help = 'config file')
    args = parser.parse_args()
    main(args)