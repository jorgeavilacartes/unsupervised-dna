# https://discuss.streamlit.io/t/how-to-use-the-key-field-in-interactive-widgets-api/1007/6
import numpy as np
from pathlib import Path
import tensorflow as tf
import streamlit as st

from unsupervised_dna import (
    ModelLoader,
    LoadImageVAE,
)

from PIL import Image

loader = ModelLoader()

@st.cache
def load_model():
    model = loader(MODEL_NAME) # load model
    model.load_weights(PATH_CKPT) # load weights
    return models

def array2img(array):
    "Array to PIL image"
    m, M = array.min(), array.max()
    # rescale to [0,1]
    img_rescaled = (array - m) / (M-m) 
    
    # invert colors black->white
    img_array = np.ceil(255 - img_rescaled*255)
    img_array = np.array(img_array, dtype="uint8")
    
    # convert to Image 
    img_pil = Image.fromarray(img_array,'L')
    return img_pil

# -- settings -- 
KMER = 8
MODEL_NAME = f"vae_{KMER}mer"
PATH_CKPT  = "checkpoint/cp.ckpt"
IMG_HEIGHT,IMG_WIDTH = 2**KMER, 2**KMER
DATA_DIR = Path("data/fcgr")
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 1

# -- load model -- 
model = load_model()    

# -- dataset --
img_loader = LoadImageVAE(IMG_HEIGHT, IMG_WIDTH)

# load path to images in /data
list_ds = tf.data.Dataset.list_files(str(DATA_DIR/'*.jpg'))
ds = list_ds.map(img_loader, num_parallel_calls=AUTOTUNE)

# Preprocessing 
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
ds = ds.map(lambda x, y: (normalization_layer(x), normalization_layer(y)))

ds_iter = iter(ds)

# Stremalit APP
st.title("Encoding DNA with FCGR and VAE")
#path_fcgr = st.sidebar.text_input(label="Path to FCGR.jpg", key="path_fcgr")
st.sidebar.button(label="predict-next", key="ta_submit")

