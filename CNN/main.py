#coding:utf8
from config import opt
import os
import time
import torch as t
import models
from data.dataset import Apks
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import random
import sys
import pynvml

def print_mem_used(line):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    f = open("Mem.txt", "a+")
    f.write("line: " + str(line) + " Mem Used: " + str(info.used / 1024 / 1024) + "\n")
    f.close()

def test(**kwargs):
    with t.no_grad():
        opt._parse(kwargs)

        # configure model
        model = getattr(models, opt.model)().eval()
        if opt.load_model_path:
            model.load(opt.load_model_path)
        model.to(opt.device)

        # data
        test_data = Apks(opt.test_data_root,test=True)
        test_dataloader = DataLoader(test_data,batch_size=opt.batch_size,
                                     shuffle=False,num_workers=opt.num_workers)
        precision, recall, f1, test_accuracy = meter_report(model, test_dataloader)
        info_str = ('test_dataset:{test_dataset},precision:{precision:.5f},recall:{recall:.5f},f1:{f1:.5f},'\
                    'test_accuracy:{test_accuracy:.5f}'.format(test_dataset=test_data.name,precision=precision,
                    recall=recall,f1=f1,test_accuracy=test_accuracy))
        print(info_str)
    
def train(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env,port = opt.vis_port)
    int_seed = random.randint(1,20)

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)
    
    # step2: data
    train_data = Apks(opt.train_data_root,train=True,int_seed=int_seed)
    val_data = Apks(opt.train_data_root,train=False,int_seed=int_seed)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    
    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)
    
    # step4: meters
    loss_meter = meter.AverageValueMeter()
    #confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10
    val_accuracy_max = opt.val_accuracy_max

    # train
    for epoch in range(opt.max_epoch):
        
        loss_meter.reset()
        #confusion_matrix.reset()

        for ii,(data,label) in enumerate(tqdm(train_dataloader)):

            # train model 
            input = data.to(opt.device)
            optimizer.zero_grad()
            
            score = model(input)
            del input
            t.cuda.empty_cache()
            
            target = label.to(opt.device)
            loss = criterion(score,target)
            
            del score, target
            t.cuda.empty_cache()
            
            loss.backward()
            optimizer.step()
            
            # meters update and visualize
            loss_meter.add(loss.item())
            del loss
            t.cuda.empty_cache()
            # detach 一下更安全保险
            #confusion_matrix.add(score.detach(), target.detach()) 

            if (ii + 1)%opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])
                
                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()
                    
        # validate and visualize
        precision, recall, f1, val_accuracy = meter_report(model,val_dataloader)

        vis.plot('val_accuracy',val_accuracy)
        info_str = ('epoch:{epoch},lr:{lr},train_loss:{loss:.6f},precision:{precision:.5f},'\
                    'recall:{recall:.5f},f1:{f1:.5f},val_accuracy:{val_accuracy:.5f}'.format(epoch=epoch,
                    loss=loss_meter.value()[0],lr=lr,precision=precision,
                    recall=recall,f1=f1,val_accuracy=val_accuracy))
        vis.log(info_str)
        # save model
        if val_accuracy > val_accuracy_max:
            prefix = ('./checkpoints/' + 'model_' + model.name + 
                      '_dataset_' + train_data.name)
            prefix += ('_epoch_{epoch}_lr_{lr}_train_loss_{loss:.6f}_precision_{precision:.5f}_'\
                       'recall_{recall:.5f}_f1_{f1:.5f}_val_accuracy_{val_accuracy:.5f}'.format(epoch=epoch,
                       loss=loss_meter.value()[0],lr=lr,precision=precision,
                       recall=recall,f1=f1,val_accuracy=val_accuracy))
            name = time.strftime(prefix + '_time_%m%d_%H-%M-%S.pth')
            model.save(name)
            val_accuracy_max = val_accuracy
        
        # update learning rate
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        previous_loss = loss_meter.value()[0]


def meter_report(model, dataloader, is_val=True):
    """
    计算模型在数据集上的精确率，召回率，f1指标，准确率
    """
    with t.no_grad():
        model.eval()
        correct_num, sum_num = 0, 0
        preds = np.ones(len(dataloader.dataset), dtype=int)
        labels = np.zeros(len(dataloader.dataset), dtype=int)
        
        for ii, (input, label) in enumerate(tqdm(dataloader)):
            
            input = input.to(opt.device)
            score = model(input)
            del input
            t.cuda.empty_cache()
            
            pred_np = t.max(score, 1)[1].detach().squeeze().cpu().numpy()
            del score
            t.cuda.empty_cache()
            
            label_np = label.detach().cpu().numpy()
            preds[sum_num:(sum_num + pred_np.size)] = pred_np
            labels[sum_num:(sum_num + pred_np.size)] = label_np 
            correct_num += (pred_np == label_np).astype(int).sum()
            sum_num += label.size(0)
            
            del label, pred_np, label_np
            t.cuda.empty_cache()
            
        if is_val:
            model.train()
        
        #report = metrics.classification_report(labels, preds)
        #print(report)
        precision = metrics.precision_score(labels, preds)
        recall = metrics.recall_score(labels, preds)
        f1 = metrics.f1_score(labels, preds)
        accuracy = metrics.accuracy_score(labels, preds)
        
        return precision, recall, f1, accuracy

def help():
    """
    打印帮助的信息： python file.py help
    """
    
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    import fire
    fire.Fire()
    
