import os
import os.path as osp
import random
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from src.args import parse_args
args = parse_args()

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def random_seed(repetition):
    if args.dataset == "Cora" or args.dataset == "CiteSeer":
        return int(repetition*10+1)
    elif args.dataset == "PubMed" or args.dataset == "Computers":
        return int(repetition)

def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)



def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        st_ = "{}_{}_".format(name, val)
        st += st_

    return st[:-1]



def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals



def compute_accuracy(preds, labels, train_mask, val_mask, test_mask):

    train_preds = preds[train_mask]
    val_preds = preds[val_mask]
    test_preds = preds[test_mask]
    train_pred_list = train_preds.cpu().numpy()
    val_preds_list = val_preds.cpu().numpy()
    test_preds_list = test_preds.cpu().numpy()
    y_train_true = labels[train_mask].cpu().numpy()
    y_val_true = labels[val_mask].cpu().numpy()
    y_test_true = labels[test_mask].cpu().numpy()


    train_acc = (torch.sum(train_preds == labels[train_mask])).float() / ((labels[train_mask].shape[0]))
    val_acc = (torch.sum(val_preds == labels[val_mask])).float() / ((labels[val_mask].shape[0]))
    test_acc = (torch.sum(test_preds == labels[test_mask])).float() / ((labels[test_mask].shape[0]))


    train_bacc = balanced_accuracy_score(y_train_true,train_pred_list)
    val_bacc = balanced_accuracy_score(y_val_true,val_preds_list)
    test_bacc = balanced_accuracy_score(y_test_true,test_preds_list)

    train_f1 = f1_score(y_train_true,train_pred_list, average='macro')
    val_f1 =  f1_score(y_val_true,val_preds_list, average='macro')
    test_f1 = f1_score(y_test_true,test_preds_list, average='macro')

    train_acc = train_acc * 100
    val_acc = val_acc * 100
    test_acc = test_acc * 100

    train_bacc = train_bacc *100
    val_bacc = val_bacc * 100
    test_bacc = test_bacc * 100

    train_f1 = train_f1 * 100
    val_f1 = val_f1 * 100
    test_f1 = test_f1 * 100

    return train_acc, val_acc, test_acc,train_bacc,val_bacc,test_bacc,train_f1,val_f1,test_f1




# def masking(fold, data):
#
#         return train_mask = data.train_mask ; val_mask = data.val_mask ; test_mask = data.test_mask





def compute_representation(net, data, device):

    net.eval()
    reps = []

    data = data.to(device)
    with torch.no_grad():
        reps.append(net(data))

    reps = torch.cat(reps, dim=0)

    return reps

