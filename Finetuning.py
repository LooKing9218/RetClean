import torch
import os
import tqdm
import torch.nn as nn
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import utils.utils as u
from utils.config import DefaultConfig
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from dataprepare.dataloader import get_data_list
import numpy as np

from models import models_vit
from util.pos_embed import interpolate_pos_embed
from dataprepare.dataloader import DatasetTrainSplitVal


# loss function
def KL(alpha, c, device):
    beta = torch.ones((1, c)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step, device):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c, device)
    return (A + B)




def val(val_dataloader, model, epoch, args, mode, device):

    print('\n')
    print('====== Start {} ======!'.format(mode))
    model.eval()
    labels = []
    outputs = []

    predictions = []
    gts = []
    correct = 0.0

    num_total = 0
    tbar = tqdm.tqdm(val_dataloader, desc='\r')

    with torch.no_grad():
        for i, img_data_list in enumerate(tbar):
            Fundus_img = img_data_list[0].to(device)
            cls_label = img_data_list[1].long().to(device)
            pred = model.forward(Fundus_img)
            evidences = [F.softplus(pred)]

            alpha = dict()
            alpha[0] = evidences[0] + 1

            S = torch.sum(alpha[0], dim=1, keepdim=True)
            E = alpha[0] - 1
            b = E / (S.expand(E.shape))

            u = args.num_classes / S


            pred = torch.softmax(b,dim=1)

            data_bach = pred.size(0)
            num_total += data_bach
            one_hot = torch.zeros(data_bach, args.num_classes).to(device).scatter_(1, cls_label.unsqueeze(1), 1)
            pred_decision = pred.argmax(dim=-1)
            for idx in range(data_bach):
                outputs.append(pred.cpu().detach().float().numpy()[idx])
                labels.append(one_hot.cpu().detach().float().numpy()[idx])
                predictions.append(pred_decision.cpu().detach().float().numpy()[idx])
                gts.append(cls_label.cpu().detach().float().numpy()[idx])
    epoch_auc = metrics.roc_auc_score(labels, outputs)
    Acc = metrics.accuracy_score(gts, predictions)

    precision = metrics.precision_score(gts, predictions,average="macro")

    recall = metrics.recall_score(gts, predictions,average="macro")

    f1 = metrics.f1_score(gts, predictions,average="macro")
    if not os.path.exists(os.path.join(args.save_model_path, "{}".format(args.net_work))):
        os.makedirs(os.path.join(args.save_model_path, "{}".format(args.net_work)))

    with open(os.path.join(args.save_model_path,"{}/{}_Metric.txt".format(args.net_work,args.net_work)),'a+') as Txt:
        Txt.write("Epoch {}: {} == Acc: {}, AUC: {}, Precession: {}, Sensitivity: {}, F1: {}\n".format(
            epoch,mode, round(Acc,6),round(epoch_auc,6),round(precision,6),round(recall,6),round(f1,6)
        ))
    print("Epoch {}: {} == Acc: {}, AUC: {}, Precession: {}, Sensitivity: {}, F1: {}\n".format(
            epoch,mode, round(Acc,6),round(epoch_auc,6),round(precision,6),round(recall,6),round(f1,6)
        ))
    torch.cuda.empty_cache()
    return epoch_auc,Acc
def train(train_loader, val_loader, test_loader, model, optimizer, criterion,writer,args,device):
    step = 0
    best_Acc = 0.0
    for epoch in range(0,args.num_epochs+1):
        model.train()
        labels = []
        outputs = []
        tq = tqdm.tqdm(total=len(train_loader) * args.batch_size)
        tq.set_description('Epoch %d, lr %f' % (epoch, args.lr))
        loss_record = []
        train_loss = 0.0
        for i, img_data_list in enumerate(train_loader):
            Fundus_img = img_data_list[0].to(device)
            cls_label = img_data_list[1].long().to(device)
            optimizer.zero_grad()
            pretict = model(Fundus_img)
            evidences = [F.softplus(pretict)]
            loss_un = 0
            alpha = dict()
            alpha[0] = evidences[0] + 1

            S = torch.sum(alpha[0], dim=1, keepdim=True)
            E = alpha[0] - 1
            b = E / (S.expand(E.shape))


            Tem_Coef = epoch*(0.99/args.num_epochs)+0.01

            loss_CE = criterion(b/Tem_Coef, cls_label)


            loss_un += ce_loss(cls_label, alpha[0], args.num_classes, epoch, args.num_epochs, device)
            loss_ACE = torch.mean(loss_un)
            loss = loss_CE+loss_ACE#
            loss.backward()
            optimizer.step()
            tq.update(args.batch_size)
            train_loss += loss.item()
            tq.set_postfix(loss='%.6f' % (train_loss / (i + 1)))
            step += 1
            one_hot = torch.zeros(pretict.size(0), args.num_classes).to(device).scatter_(1, cls_label.unsqueeze(1), 1)
            pretict = torch.softmax(pretict, dim=1)
            for idx_data in range(pretict.size(0)):
                outputs.append(pretict.cpu().detach().float().numpy()[idx_data])
                labels.append(one_hot.cpu().detach().float().numpy()[idx_data])

            if step%10==0:
                writer.add_scalar('Train/loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        torch.cuda.empty_cache()
        loss_train_mean = np.mean(loss_record)
        epoch_train_auc = metrics.roc_auc_score(labels, outputs)

        del labels,outputs

        writer.add_scalar('Train/loss_epoch', float(loss_train_mean),
                          epoch)
        writer.add_scalar('Train/train_auc', float(epoch_train_auc),
                          epoch)

        print('loss for train : {}, {}'.format(loss_train_mean,round(epoch_train_auc,6)))
        if epoch % args.validation_step == 0:
            if not os.path.exists(os.path.join(args.save_model_path, "{}".format(args.net_work))):
                os.makedirs(os.path.join(args.save_model_path, "{}".format(args.net_work)))
            with open(os.path.join(args.save_model_path, "{}/{}_Metric.txt".format(args.net_work,args.net_work)), 'a+') as f:
                f.write('EPOCH:' + str(epoch) + ',')


            mean_AUC, mean_ACC = val(val_loader, model, epoch,args,mode="val",device=device)
            writer.add_scalar('Valid/Mean_val_AUC', mean_AUC, epoch)
            is_best = mean_ACC > best_Acc
            if is_best:
                best_Acc = max(best_Acc, mean_ACC)
                mean_AUC_test, mean_ACC_test = val(test_loader, model, epoch,
                                                   args, mode="test", device=device)


                checkpoint_dir = os.path.join(args.save_model_path)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                print('===> Saving models...')
                u.save_checkpoint_epoch({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'mean_AUC': mean_AUC,
                    'mean_ACC': mean_ACC,
                }, mean_AUC, mean_ACC,mean_AUC_test, mean_ACC_test,epoch, True, checkpoint_dir, stage="Test",
                    filename=os.path.join(checkpoint_dir,"checkpoint.pth.tar"))


def main(args=None,writer=None):

    train_files = get_data_list(args.root)
    train_dataset, val_dataset = train_test_split(train_files,
                                                   test_size=0.2,
                                                   random_state=args.seed,
                                                   shuffle=True)



    train_loader = DataLoader(DatasetTrainSplitVal(data_list=train_dataset,mode='train'),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)

    val_loader = DataLoader(DatasetTrainSplitVal(data_list=val_dataset,mode='val'),
        batch_size=args.batch_size, shuffle=False, pin_memory=True)


    test_dataset = get_data_list(args.root_test)

    test_loader = DataLoader(DatasetTrainSplitVal(data_list=test_dataset, mode='test'),
               batch_size=args.batch_size, shuffle=False, pin_memory=True)



    # bulid model
    device = torch.device('cuda:{}'.format(args.cuda))
    # call the model
    model = models_vit.__dict__['vit_large_patch16'](
        num_classes= args.num_classes,
        drop_path_rate=0.2,
        global_pool=True,
    )

    # load RETFound weights
    checkpoint = torch.load(
        'PathTo/RETFound_cfp_weights.pth',
        map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)
    # load pre-trained model
    model.load_state_dict(checkpoint_model, strict=False)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # set the loss criterion
    criterion = nn.CrossEntropyLoss().to(device)
    train(train_loader, val_loader,test_loader, model, optimizer, criterion,writer,args,device)

if __name__ == '__main__':
    torch.set_num_threads(1)
    args = DefaultConfig()
    args.seed = 1234
    u.setup_seed(args.seed)
    log_dir = os.path.join(args.log_dirs,"Seed_{}".format(args.seed))
    writer = SummaryWriter(log_dir=log_dir)
    args.save_model_path = os.path.join(args.save_model_path,"Seed_{}".format(args.seed))

    main(args=args, writer=writer)