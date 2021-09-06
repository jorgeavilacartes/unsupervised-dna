# https://discuss.streamlit.io/t/how-to-use-the-key-field-in-interactive-widgets-api/1007/6
from pathlib import Path
import tensorflow as tf
import streamlit as st

from unsupervised_dna import (
    ModelLoader,
    LoadImageVAE,
)

# -- settings -- 
KMER = 8
MODEL_NAME = f"vae_{KMER}mer"
PATH_CKPT  = "checkpoint/cp.ckpt"
IMG_HEIGHT,IMG_WIDTH = 2**KMER, 2**KMER
DATA_DIR = Path("data")
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 1

# -- load model -- 
loader = ModelLoader()

@st.cache
def load_model():
    model = loader(MODEL_NAME) # load model
    model.load_weights(PATH_CKPT) # load weights
    return model

model = load_model()    

# -- dataset --
img_loader = LoadImageVAE(IMG_HEIGHT, IMG_WIDTH)

# load path to images in /data
list_ds = tf.data.Dataset.list_files(str(DATA_DIR/'*.jpg'))
ds = list_ds.map(img_loader, num_parallel_calls=AUTOTUNE)

# Performance of datasets
def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


# Stremalit APP
st.title("Encoding DNA with FCGR and VAE")
#path_fcgr = st.sidebar.text_input(label="Path to FCGR.jpg", key="path_fcgr")
st.sidebar.button(label="predict-next", key="ta_submit")

it = iter(ds)