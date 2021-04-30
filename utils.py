import numpy as np
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import datasets.labels as l

def convertColour(inp, label):
    np_inp = inp.numpy()
    label = label.numpy()
    final = []
    for i, i_label in zip(np_inp, label):
        out = []
        for j, j_label in zip(i, i_label):
            if j_label == 255:
                out.append(l.trainId2color[j_label])
            else:
                out.append(l.trainId2color[j])
        final.append(out)
    return np.asarray(final)

# IOU function
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, device): 
    SMOOTH = 1e-6
    
    # output shape: (1,20,1028,2056), label shape: (20,1028,2056)
    iou_class={} 
    num_class = list(outputs.size())[1]
    for i in range(20): 
        # get the most likely class out of the probabilities calculated for each class
        output_class_most_likely, indices =torch.max(outputs,dim=1)
        mask_size = list(labels.size())[1:]
        boolean_mask = torch.ones(mask_size, dtype=torch.float64, device = device).bool()
        # create a mask for each class
        mask = torch.ones(mask_size, dtype=torch.float64, device = device)*i

        select_label_class = torch.eq(mask, labels).to(device)
        select_predicted_class = torch.eq(mask, indices).to(device)
        intersection = (select_label_class & select_predicted_class)
        # union is number of predicted samples and number of labelled samples for each class - intersection
        union_sum = select_predicted_class.sum()+select_label_class.sum() - intersection.sum()  
        
        iou = (intersection.sum() + SMOOTH) / (union_sum + SMOOTH) 
        iou_class[i]=iou
    return sum(list(iou_class.values()))/num_class

def save_checkpoint(model, loss_list,val_loss_list,train_iou,val_iou,batch_size,epoch,lr,optimizer_name, use_psp):
    
    if use_psp:
        model.name = "batch_"+str(batch_size)+"_lr_"+str(lr)+"_e_"+str(epoch)+"_optimizer_"+optimizer_name+"_psp"
        use_psp_str = "True"
        dirName = "./Models/B"+str(batch_size)+"L"+str(lr)+str(optimizer_name)+"_psp"
    else:
        model.name = "batch_"+str(batch_size)+"_lr_"+str(lr)+"_e_"+str(epoch)+"_optimizer_"+optimizer_name
        use_psp_str = "False"
        dirName = "./Models/B"+str(batch_size)+"L"+str(lr)+str(optimizer_name)
        
    model.batch_size = batch_size
    model.epoch = epoch
    model.lr = lr
    model.optimizer_name = optimizer_name
    model.use_psp = use_psp
    model.loss_list = loss_list
    model.val_loss_list = val_loss_list
    model.train_iou = train_iou
    model.val_iou = val_iou
    
    checkpoint = {'batch_size': model.batch_size,
              'epoch': model.epoch,
              'lr': model.lr,
              'optimizer_name':model.optimizer_name,
              'use_psp': use_psp_str,
              'loss_list': model.loss_list,
              'val_loss_list': model.val_loss_list,
              'train_iou': model.train_iou,
              'val_iou': model.val_iou,
              'model_name':model.name,
              'state_dict': model.state_dict(),    
              }
    
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    
    torch.save(checkpoint, dirName+"/"+model.name+".pt")

def load_model(path,device):
    
    from models import BigDLModel
    
    cp = torch.load(path)
    
#     if cp['use_psp']=="True":
#         model.name = "batch_"+str(batch_size)+"_lr_"+str(lr)+"_e_"+str(epoch)+"_optimizer_"+optimizer_name+"_psp"
#     else:
#         model.name = "batch_"+str(batch_size)+"_lr_"+str(lr)+"_e_"+str(epoch)+"_optimizer_"+optimizer_name
        
    model = BigDLModel(use_psp=True).to(device)
    
    model.name = cp['model_name']
    model.batch_size = cp['batch_size']
    model.epoch = cp['epoch']
    model.lr = cp['lr']
    model.use_psp = cp['use_psp']
    model.loss_list = cp['loss_list']
    model.val_loss_list = cp['val_loss_list']
    model.train_iou = cp['train_iou']
    model.val_iou = cp['val_iou']
    model.load_state_dict(cp['state_dict'])
    model.optimizer_name = cp['optimizer_name']
   
    

    return model

def get_optimizer(optimizer_name,model,lr):
    switcher={
            "adam":torch.optim.Adam(model.parameters(),weight_decay = 1e-5, lr = lr),
            "rmsprop": torch.optim.RMSprop(model.parameters(),weight_decay = 1e-5, lr = lr),
            "sgd": torch.optim.SGD(model.parameters(),weight_decay = 1e-5, lr = lr)
         }
    return switcher.get(optimizer_name,"Invalid optimizer name")
   