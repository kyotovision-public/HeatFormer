import os
import torch
import numpy
import logging
import os.path as osp

from lib.core.config import ROOT_DIR

def create_logger(logdir, phase='train'):
    os.makedirs(osp.join(ROOT_DIR, logdir), exist_ok=True)

    log_file = osp.join(ROOT_DIR, logdir, f'{phase}_log.txt')

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_file,
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def move_dict_to_device(dic, device, tensor2float=True):
    try:
        for k,v in dic.items():
            if k=='img_name':continue
            if isinstance(v, torch.Tensor):
                if tensor2float:
                    dic[k] = v.float().to(device)
                else:
                    dic[k] = v.to(device)
            elif isinstance(v, dict):
                move_dict_to_device(v, device)
            elif isinstance(v, list):
                for i in range(len(v)):
                    if type(v[i])==tuple:
                        continue
                    elif type(v[i])==list:
                        continue
                    elif type(v[i])==numpy.str_:
                        continue
                    elif type(v[i])==str:
                        continue
                    else:
                        if tensor2float:
                            dic[k] = v[i].float().to(device)
                        else:
                            dic[k] = v[i].to(device)
    except:
        print(f'error key : {k}')
        import pdb;pdb.set_trace()

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_optimizer(model, optim_type, lr, weight_decay, momentum=None):
    if optim_type in ['sgd', 'SGD']:
        opt = torch.optim.SGD(lr=lr, params=model.parameters(), momentum=momentum)
    elif optim_type in ['Adam', 'adam', 'ADAM']:
        opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    else:
        raise ModuleNotFoundError
    return opt