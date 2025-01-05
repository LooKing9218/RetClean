import torch
import shutil
import random
import imgaug as ia
import numpy as np
import os.path as osp
import torch.backends.cudnn as cudnn

def adjust_learning_rate(opt, optimizer, epoch):
    """
    将学习速率设置为初始LR经过每30个epoch衰减10% (step = 30)
    """
    if opt.lr_mode == 'step':
        lr = opt.lr * (0.1 ** (epoch // opt.step))
    elif opt.lr_mode == 'poly':
        lr = opt.lr * (1 - epoch / opt.num_epochs) ** 0.9
    elif opt.lr_mode == 'normal':
        lr = opt.lr
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))

    for param_group in optimizer.param_groups:                                 #可以为一个网络设置多个优化器，每个优化器对应一个字典包括参数组及其对应的学习率,动量等等，optimizer.param_groups是由所有字典组成的列表
        param_group['lr'] = lr                                                 #动态修改学习率
    return lr
def save_checkpoint(state,best_pred,best_pred_Test, epoch,is_best,checkpoint_path,stage="val",filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("Model Saving................")
        shutil.copyfile(filename, osp.join(checkpoint_path,'model_{}_{:03d}_{:.6f}_{:.6f}.pth.tar'.format(
            stage,(epoch + 1),best_pred,best_pred_Test)))

def save_checkpoint_epoch(state,pred_Auc,pred_ACC,mean_AUC_test, mean_ACC_test,epoch,is_best,checkpoint_path,stage="val",filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("Model Saving................")
        shutil.copyfile(filename, osp.join(checkpoint_path,'model_{}_{:03d}_Val_{:.6f}_{:.6f}_Test_{:.6f}_{:.6f}.pth.tar'.format(
            stage,(epoch + 1),pred_Auc,pred_ACC,mean_AUC_test, mean_ACC_test)))


def setup_seed(seed=1234):
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # torch.cuda.manual_seed_all()为多个GPU设置种子
    np.random.seed(seed)
    random.seed(seed)
    ia.seed(seed)

    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True