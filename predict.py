import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image

from tqdm import tqdm
from PIL import Image
import numpy as np
import os

from utils import convertColour, save_checkpoint, load_model, iou_pytorch, get_optimizer
from datasets.cityscapes import setupDatasetsAndLoaders
from datasets.labels import label2trainid

def evaluate(dir, model_path, batch_size=1, gpu=True):
    device = "cuda" if (torch.cuda.is_available() and gpu) else "cpu"

    model = load_model(model_path, device)
    test_loader = setupDatasetsAndLoaders(dir, batch_size)[-1]
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    # Reset loss
    running_val_loss = 0
    running_val_iou = 0

    # Run predictions on validation set
    for data_val in tqdm(test_loader):
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
        
    val_loss = running_val_loss/len(test_loader.dataset)
    val_iou = running_val_iou/len(test_loader.dataset)
    print(f'Evaluating {model.name} at epoch {model.epoch}')
    print(f'val_loss: {val_loss}\nval_iou: {val_iou}')

# evaluate('./data', './Models/B1L0.001adam/batch_1_lr_0.001_e_1_optimizer_adam.pt',1,True)
def predict(image_paths, model_path, gpu=True):
    device = "cuda" if (torch.cuda.is_available() and gpu) else "cpu"
    out_path = image_paths[-1]
    seq_paths = image_paths[:4]

    model = load_model(model_path, device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224,448)),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std),
    ])

    images = []
    for path in seq_paths:
        image = Image.open(path).convert('RGB')
        image = transform(image).unsqueeze(0)
        
        images.append(image)

    annotatedOutput = Image.open(out_path)
    annotatedOutput = transforms.Resize((224,448))(annotatedOutput)
    annotatedOutput = np.array(annotatedOutput)

    temp = annotatedOutput.copy()
    # converting labelIDs to trainIDs
    for k,v in label2trainid.items():
        temp[annotatedOutput == k] = v
    
    annotatedOutput = torch.from_numpy(temp).unsqueeze(0)

    for ind in range(4):
        images[ind] = Variable(images[ind]).to(device)

    # Forward pass
    output = model(images)
    annotatedOutput = annotatedOutput.to(device).long()

    val_loss = loss_fn(output, annotatedOutput)
    val_iou_mean = iou_pytorch(output, annotatedOutput,device)
    print(f'Predicting using {model.name} at epoch {model.epoch}')
    print(f'val_loss: {val_loss}\nval_iou: {val_iou_mean}')
    return output, val_iou_mean


def retrieve_sequence(img_path, data_dir='data_display'):
    img_name = img_path.split('_')
    seq = int(img_name[1])
    ref_frame = int(img_name[2])
    paths = []
    for frame in range(ref_frame-3, ref_frame+1):
        path = '{}_{:06d}_{:06d}_leftImg8bit.png'.format(img_name[0], seq, frame)
        paths.append(os.path.join(data_dir, 'leftImg8bit_sequence', path))
    
    annotated_path = '{}_{:06d}_{:06d}_gtFine_labelIds.png'.format(img_name[0], seq, frame)
    paths.append(os.path.join(data_dir, 'gtFine', annotated_path))
    
    return paths

# seq = retrieve_sequence('ulm_000015_000019_leftImg8bit.png')
# predict(seq, 'Models/B1L0.001adam_psp/batch_1_lr_0.001_e_1_optimizer_adam_psp.pt', True)
# model_path = 'Models/B4L0.001adam_psp/batch_4_lr_0.001_e_9_optimizer_adam_psp.pt'
# model_path = 'Models/B1L0.001adam_psp/batch_1_lr_0.001_e_1_optimizer_adam_psp.pt'
# evaluate('./data', model_path, 4, False)
