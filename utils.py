import numpy as np
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

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

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, smooth = 1e-6):   
    iou_class={}
    for i in range(19):
        output_class_h = outputs[0][i] 
        output_class = torch.unsqueeze(output_class_h,0)
        output_class= output_class.type(labels.dtype)

        intersection = (output_class & labels).float().sum((1, 2))
        union = (output_class | labels).float().sum((1, 2))  
        
        iou = (intersection + smooth) / (union + smooth) 
        iou_class[i]=iou
    return sum(list(iou_class.values()))/19 

def save_checkpoint(model, loss_list,val_loss_list,train_iou,val_iou,batch_size,epoch,lr,optimizer_name, use_psp):
    
    if use_psp:
        model.name = "batch_"+str(batch_size)+"_lr_"+str(lr)+"_e_"+str(epoch)+"_optimizer_"+optimizer_name+"_psp"
    else:
        model.name = "batch_"+str(batch_size)+"_lr_"+str(lr)+"_e_"+str(epoch)+"_optimizer_"+optimizer_name
        
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
              'use_psp': use_psp,
              'loss_list': model.loss_list,
              'val_loss_list': model.val_loss_list,
              'train_iou': model.train_iou,
              'val_iou': model.val_iou,
              'model_name':model.name,
              'state_dict': model.state_dict(),    
              }
    
    
    torch.save(checkpoint, "./Models/"+model.name+".pt")
    
def load_model(path):
    cp = torch.load(path)
    if cp.use_psp:
        model = BigDLModel(use_psp)
        
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
   