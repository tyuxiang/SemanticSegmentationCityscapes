from datasets.cityscapes import setupDatasetsAndLoaders
from models import BigDLModel
from utils import convertColour, save_checkpoint, load_model, iou_pytorch, get_optimizer
from tqdm import tqdm
import itertools
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

def train(batch_size =1, num_epochs= 20, lr = 0.001, optimizer_name = "adam", use_psp= False):
    
    # Get data
    train_set, val_set, test_set, train_loader, val_loader, test_loader = setupDatasetsAndLoaders('./data', batch_size=batch_size)
    
    # Set device
    CUDA = True
#     device = "cuda" if (torch.cuda.is_available() and CUDA) else "cpu"
    device = "cpu"
    print("Using device:",device)
    print("Torch version:",torch.__version__)
    
    # Our model
    model = BigDLModel(use_psp).to(device)
    
    # Set loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = get_optimizer(optimizer_name, model, lr)
    
    # Lists for later
    loss_list = []
    val_loss_list = []
    train_iou = []
    val_iou = []
    
    
    for epoch in range(num_epochs):
        
        # Reset loss
        running_train_loss = 0
        running_val_loss = 0
        running_train_iou = 0
        running_val_iou = 0
        
        # Set to training mode
        model.train()
        
        # Train the model for one epoch
        for data in tqdm(train_loader):
            
            # Get data
            imgSeq, annotatedOutput, imgName = data

            # Send data to device
            for ind in range(4):
                imgSeq[ind] = Variable(imgSeq[ind]).to(device)

            # Forward pass
            output = model(imgSeq)
            
            # Calculate loss and IOU
            annotatedOutput = annotatedOutput.to(device).long()
            loss = loss_fn(output, annotatedOutput)
            running_train_loss+=loss.item()*output.shape[0]
            iou_mean = iou_pytorch(output, annotatedOutput,device)
            running_train_iou+=iou_mean.item()*output.shape[0]

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Switch to evaluation
        model.eval()
        
        # Run predictions on validation set
        for data_val in val_loader:
           
            # Get data
            imgSeq, annotatedOutput, imgName = data_val
            
            # Send to device
            for ind in range(4):
                imgSeq[ind] = Variable(imgSeq[ind]).to(device)
                
            # Forward pass
            output = model(imgSeq)
            
            # Calculate loss and IOU
            annotatedOutput = annotatedOutput.to(device).long()
            val_loss = loss_fn(output, annotatedOutput)
            val_iou_mean = iou_pytorch(output, annotatedOutput,device)
            running_val_loss+=val_loss.item()*output.shape[0]
            running_val_iou+=val_iou_mean.item()*output.shape[0]
            
        # Display
        print('epoch {}/{}, loss {:.4f}, val_loss {:.4f}, train IoU {:.4f}, val IoU {:.4f}'.format(
            epoch + 1, num_epochs, 
            running_train_loss/len(train_loader.dataset), 
            running_val_loss/len(val_loader.dataset),
            running_train_iou/len(train_loader.dataset), 
            running_val_iou/len(val_loader.dataset)))

        loss_list.append(running_train_loss/len(train_loader.dataset))
        val_loss_list.append(running_val_loss/len(val_loader.dataset))
        train_iou.append(running_train_iou/len(train_loader.dataset))
        val_iou.append(running_val_iou/len(val_loader.dataset))

        # Save model here
        save_checkpoint(model, loss_list,val_loss_list,train_iou,val_iou,batch_size,epoch,lr,optimizer_name,use_psp)
        
    return model, val_iou
        
def hyperparams_train(optimizer="adam",use_psp=True):
    
    # Hyperparameters to train over
    batch_sizes = [8, 16, 32]
    lrates = [0.001, 0.005, 0.0001, 0.0005]
    max_iou = 0
    best_model = "None"
    
    # Iterate through hyperparams to find best
    for batch_size, lr in itertools.product(batch_sizes, lrates):
        print("Training: Batchsize =",batch_size,",Learning Rate =",lr)
        model, iou = train(num_epochs=20, optimizer_name = optimizer, lr= lr, batch_size = batch_size,use_psp = use_psp)
        if max(iou) > max_iou:
            max_iou = max(iou)
            best_model = model.name
            print("Best model so far:", best_model)
    print("Finished training and best model is",best_model)
    
# def best_params():
    
#     device = "cpu"
#     batch_sizes = [8]
#     lrates = [0.001, 0.005, 0.0001]
#     loss_fn = nn.CrossEntropyLoss(ignore_index=255)
#     max_iou = 0
#     best_model = "None"
    
#     # Iterate through hyperparams to find best
#     for batch_size, lr in itertools.product(batch_sizes, lrates):
        
#         # Get data
#         train_set, val_set, test_set, train_loader, val_loader, test_loader = setupDatasetsAndLoaders('./data', batch_size=batch_size)
        
#        # Lists for later
#         loss_list = []
#         val_loss_list = []
#         train_iou = []
#         val_iou = []
        
#         for epoch in range(20):
#             print("Testing: Batchsize =",batch_size,",Learning Rate =",lr,"Epoch =",epoch)
            
#             # Reset loss
#             running_train_loss = 0
#             running_val_loss = 0
#             running_train_iou = 0
#             running_val_iou = 0
            
            
#             # Get model
#             path = "Models/B"+str(batch_size)+"L"+str(lr)+"ADAMPSP/batch_"+str(batch_size)+"_lr_"+str(lr)+"_e_"+str(epoch)+"_optimizer_adam_psp.pt"
#             model = load_model(path,"cpu")
            

#             # Run predictions on validation set
#             for data_val in val_loader:

#                 # Get data
#                 imgSeq, annotatedOutput, imgName = data_val

#                 # Send to device
#                 for ind in range(4):
#                     imgSeq[ind] = Variable(imgSeq[ind]).to(device)

#                 # Forward pass
#                 output = model(imgSeq)

#                 # Calculate loss and IOU
#                 annotatedOutput = annotatedOutput.to(device).long()
#                 val_loss = loss_fn(output, annotatedOutput)
#                 val_iou_mean = iou_pytorch(output, annotatedOutput,device)
#                 running_val_loss+=val_loss.item()*output.shape[0]
#                 running_val_iou+=val_iou_mean.item()*output.shape[0]
                
#             loss_list.append(running_train_loss/len(train_loader.dataset))
#             val_loss_list.append(running_val_loss/len(val_loader.dataset))
#             train_iou.append(running_train_iou/len(train_loader.dataset))
#             val_iou.append(running_val_iou/len(val_loader.dataset))
                
#             if max(val_iou) > max_iou:
#                 max_iou = max(val_iou)
#                 best_model = model.name
#                 print("Best model so far:", best_model,"with val_iou:",val_iou)
#     print("Finished training and best model is",best_model)
    

            

        
# hyperparams_train(optimizer = "adam",use_psp= True)
# train(use_psp=True)
# best_params()