
import os
import json
import argparse
import numpy as np
from shutil import copyfile
import torch
import torch.multiprocessing as mp
import random

from core.trainer_cycle import Trainer
from core.dist import (
    get_world_size,
    get_local_rank,
    get_global_rank,
    get_master_ip,
)

parser = argparse.ArgumentParser(description='STTN')
parser.add_argument('-cnf', '--config', default='configs/youtube-vos.json', type=str)
parser.add_argument('-m', '--model', default='sttn', type=str)
parser.add_argument('-p', '--port', default='23455', type=str)
parser.add_argument('-e', '--exam', action='store_true')
parser.add_argument('-c', '--ckptpath', default='/release_model/notexistant/', type=str) #added by Rema for loading initializing model
parser.add_argument('-cRem', '--ckptpathRem', default='/release_model/notexistant/', type=str) #added by Rema for loading initializing model
parser.add_argument('-cAdd', '--ckptpathAdd', default='/release_model/notexistant/', type=str) #added by Rema for loading initializing model
parser.add_argument('-cn', '--ckptnumber', default='8', type=str)
parser.add_argument('-cnRem', '--ckptnumberRem', default='8', type=str)
parser.add_argument('-cnAdd', '--ckptnumberAdd', default='8', type=str)
args = parser.parse_args()


def main_worker(rank, config):
    if 'local_rank' not in config:
        config['local_rank'] = config['global_rank'] = rank
    if config['distributed']:
        torch.cuda.set_device(int(config['local_rank']))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=config['init_method'],
                                             world_size=config['world_size'],
                                             rank=config['global_rank'],
                                             group_name='mtorch'
                                             )
        print('using GPU {}-{} for training'.format(
            int(config['global_rank']), int(config['local_rank'])))

    config['save_dir'] = os.path.join(config['save_dir'], '{}_{}'.format(config['model'],
                                                                         os.path.basename(args.config).split('.')[0]))
    if torch.cuda.is_available(): 
        config['device'] = torch.device("cuda:{}".format(config['local_rank']))
    else: 
        config['device'] = 'cpu'

    if (not config['distributed']) or config['global_rank'] == 0:
        os.makedirs(config['save_dir'], exist_ok=True)
        config_path = os.path.join(
            config['save_dir'], config['config'].split('/')[-1])
        if not os.path.isfile(config_path):
            # copyfile(config['config'], config_path)
            config_save=config.copy()
            del config_save['device']
            with open(config_path, 'w') as convert_file:
                json.dump(config_save,convert_file, indent=4)
        print('[**] create folder {}'.format(config['save_dir']))
    
    trainer = Trainer(config, debug=args.exam)
    trainer.train()


if __name__ == "__main__":
    # setting manual seed:
    torch.manual_seed(2020)
    random.seed(2020)
    np.random.seed(2020)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)
    os.environ['PYTHONHASHSEED'] = str(2020)

    # loading configs
    config = json.load(open(args.config))
    os.environ["CUDA_VISIBLE_DEVICES"]=config['gpu'] #added by Rema to choose only one GPU
    config['model'] = args.model
    config['config'] = args.config
    if 'masking' not in config['data_loader']: 
        config['data_loader']['masking']="loaded"
    print("masks are:", config['data_loader']['masking'])
    config['initialmodelA'] = args.ckptpathRem #added by Rema for loading initializing model
    config['initialmodelB'] = args.ckptpathAdd #added by Rema for loading initializing model
    config['initialmodel'] = args.ckptpath #added by Rema for loading initializing model
    config['chosen_epoch'] = args.ckptnumber
    config['chosen_epochA'] = args.ckptnumberRem
    config['chosen_epochB'] = args.ckptnumberAdd

    # setting distributed configurations
    config['world_size'] = get_world_size()
    config['init_method'] = f"tcp://{get_master_ip()}:{args.port}"
    config['distributed'] = True if config['world_size'] > 1 else False

    # setup distributed parallel training environments
    if get_master_ip() == "127.0.0.1":
        # manually launch distributed processes 
        mp.spawn(main_worker, nprocs=config['world_size'], args=(config,))
    else:
        # multiple processes have been launched by openmpi 
        config['local_rank'] = get_local_rank()
        config['global_rank'] = get_global_rank()
        main_worker(-1, config)
