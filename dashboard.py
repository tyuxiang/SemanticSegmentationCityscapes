import os
from PIL import Image
import numpy as np
import streamlit as st

from utils import iou_pytorch

st.set_page_config(layout='wide')

@st.cache(show_spinner=False)
def load_image(path):
    image = Image.open(path)
    return image

@st.cache(show_spinner=False)
def predict_image(model, path):
    img = Image.fromarray((np.random.rand(1024, 2048, 3)*255).astype(np.uint8))
    img = img.convert('RGBA')

    # iou_pytorch(outputs, labels)
    return img, 0.582

display_dir = './data_display'
all_display_samples = [img.split('_leftImg8bit')[0] for img in os.listdir(os.path.join(display_dir, 'leftImg8bit'))]




### Dashboard structure ###
st.title('Semantic Segmentation for Roads')
st.sidebar.title("About")
st.sidebar.info("This is an application written by us")

# models = load_all_models()

select_left, select_right = st.beta_columns(2)
with select_left:
    # apparently streamlit can just allow you to write text like this
    st.header("Model Selection")
    "Pick a trained model that we should predict the image with!"

    option_empty = st.empty()
    model_selected = option_empty.radio("Select model that you want", (
                                            "Model 1", 
                                            "Model 2?", 
                                            "Model 3?", 
                                            "Model 4? idk", 
                                        ))

    # uploaded_file = st.file_uploader("Or choose an image", type="jpg")

with select_right:
    st.header("Image Selection")
    "Pick an image from our existing system!"
    image_name = st.selectbox("Pick an image.", all_display_samples).split('_')
    ref_frame = int(image_name[2])
    image_paths = ['{}_{}_{:06d}_leftImg8bit.png'.format(image_name[0], image_name[1], frame) for frame in range(ref_frame-3, ref_frame+1)]


st.header('Image Sequence')
display_r1 = st.beta_columns(4)
for idx, display in enumerate(display_r1):
    with display:
        # image_box = st.empty()
        image = load_image(os.path.join(display_dir, 'leftImg8bit_sequence', image_paths[idx]))
        st.image(image, caption=f"{image_paths[idx]}", use_column_width=True)

st.header('Annotated Image')
display_r2 = st.beta_columns(2)
with display_r2[0]:
    st.subheader('Ground Truth Annotations')
    gnd_truth = '{}_{}_{}_gtFine_color.png'.format(image_name[0], image_name[1], image_name[2])
    image = load_image(os.path.join(display_dir, 'gtFine', gnd_truth))
    st.image(image, caption=f"{gnd_truth}", use_column_width=True)

with display_r2[1]:
    st.subheader('Predicted Annotations')
    prediction_image = st.empty()
    iou_box = st.empty()

    
    with st.spinner("prediction happening... please wait"):
        try:
            output, iou_score = predict_image(model_selected, os.path.join(display_dir, 'leftImg8bit_sequence', image_paths[-1]))
            prediction_image.image(output, caption="annotated image from model", use_column_width=True)
            iou_box.success(f'IOU Score is {iou_score}')

        except:
            prediction_image.error("there has been an error :(")
