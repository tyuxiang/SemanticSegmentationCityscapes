import os
from PIL import Image
import numpy as np
import streamlit as st
import pandas as pd

import torchvision.transforms as transforms
import torch

from utils import iou_pytorch, convertColour
from predict import predict, retrieve_sequence
from datasets.labels import trainId2color, label2trainid
from preprocessing.augment_script import augment_image

st.set_page_config(layout='wide')

@st.cache(show_spinner=False)
def load_image(path):
    image = Image.open(path)
    return image

# @st.cache(show_spinner=False)
def predict_image(model, path, gnd_truth_image):
    seq = retrieve_sequence(path)
    output, iou_score = predict(seq, model, True)
    output = output.squeeze(0)
    output = torch.max(output.cpu(),0).indices.float()

    gnd_truth_image = transforms.ToTensor()(transforms.Resize((224,448))(gnd_truth_image)).squeeze(0)

    img_np = convertColour(output, gnd_truth_image).astype(np.uint8)
    img = torch.Tensor(img_np)
    img = transforms.ToPILImage()(img_np)

    return img, iou_score

@st.cache()
def get_graph_data(model_path):
    cp = torch.load(model_path)
    train_loss = cp['loss_list']
    val_loss = cp['val_loss_list']
    train_iou = cp['train_iou']
    val_iou = cp['val_iou']

    n = len(train_loss)
    ran = range(1, n+1)

    return ran, train_loss, val_loss, train_iou, val_iou

@st.cache()
def get_model_list(model_dir):
    archs = os.listdir(model_dir)
    all_models = []
    for arch in archs:
        eps = [os.path.join(arch, p) for p in os.listdir(os.path.join(model_dir, arch))]
        all_models += eps
    all_models.sort()

    return all_models

display_dir = './data_display'
all_display_samples = [img.split('_leftImg8bit')[0] for img in os.listdir(os.path.join(display_dir, 'leftImg8bit'))]




### Dashboard structure ###
st.title('Semantic Segmentation for Roads')

select_left, select_right = st.beta_columns(2)
with select_left:
    st.header("Model Selection")
    "Pick a trained model that we should predict the image with!"

    option_empty = st.empty()
    model_selected = option_empty.selectbox("Select model that you want", get_model_list('./Models'))

with select_right:
    st.header("Image Selection")
    "Pick an image from our existing system!"
    image_name = st.selectbox("Pick an image.", all_display_samples).split('_')
    ref_frame = int(image_name[2])
    image_paths = ['{}_{}_{:06d}_leftImg8bit.png'.format(image_name[0], image_name[1], frame) for frame in range(ref_frame-3, ref_frame+1)]


st.sidebar.subheader('Image Sequence')
for idx in range(4):
    image = load_image(os.path.join(display_dir, 'leftImg8bit_sequence', image_paths[idx]))
    st.sidebar.image(image, caption=f"{image_paths[idx]}", use_column_width=True)

st.header('Annotated Image')
display_r2 = st.beta_columns(2)
with display_r2[0]:
    st.subheader('Ground Truth Annotations')
    gnd_truth = '{}_{}_{}_gtFine_color.png'.format(image_name[0], image_name[1], image_name[2])
    gnd_truth_image = load_image(os.path.join(display_dir, 'gtFine', gnd_truth))

    st.image(gnd_truth_image, caption=f"{gnd_truth}", use_column_width=True)

with display_r2[1]:
    st.subheader('Predicted Annotations')
    prediction_image = st.empty()
    iou_box = st.empty()

    annotation = '{}_{}_{}_gtFine_labelIds.png'.format(image_name[0], image_name[1], image_name[2])
    annotation_image = load_image(os.path.join(display_dir, 'gtFine', annotation))

    with st.spinner("prediction happening... please wait"):
        try:
            model_path = os.path.join('./Models', model_selected)
            output, iou_score = predict_image(model_path, image_paths[-1], annotation_image)
            prediction_image.image(output, caption="annotated image from model", use_column_width=True)
            iou_box.success(f'IOU Score is {iou_score}')

        except:
            prediction_image.error("there has been an error :(")

st.header('Graphs from Training')
graph_data = get_graph_data(model_path)

display_r3 = st.beta_columns(2)
with display_r3[0]:
    st.subheader('Loss Graph (across epochs)')
    d = pd.DataFrame([graph_data[1], graph_data[2]]).T
    d.columns = ['train_loss', 'val_loss']
    st.line_chart(d)

with display_r3[1]:
    st.subheader('IOU Graph (across epochs)')
    d = pd.DataFrame([graph_data[3], graph_data[4]]).T
    d.columns = ['train_iou', 'val_iou']
    st.line_chart(d)
