"Encode sequences using VAE model"
from parameters import PARAMETERS

from tqdm import tqdm
from pathlib import Path
import numpy as np
import tensorflow as tf

from unsupervised_dna import (
    ModelLoader,
    LoadImageEncoder,
    MonitorValues,
)

mv = MonitorValues(["path_img","z_mean","z_var"])

# config experiment
KMER = PARAMETERS["KMER"]
DATA_DIR = Path(f"data/fcgr-{KMER}-mer") # image directory
ENCODING_DIR = Path("encoding") # image directory
ENCODING_DIR.mkdir(exist_ok=True)
LIST_IMG = [path for path in DATA_DIR.rglob("*.jpg")] # list of images to predict
load_img = LoadImageEncoder(2**KMER,2**KMER)

# Load Encoder
loader = ModelLoader()
vae = loader(f"vae_{KMER}mer", weights_path = "checkpoint/cp.ckpt")
encoder = tf.keras.Model(inputs=vae.input,
                                outputs=vae.layers[1].output
                        )

# aux = np.random.rand(128,128,1) # grayscale image, 1 channel
# output = encoder(aux)

normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

# encode FCGR
for path_img in tqdm(LIST_IMG, desc="Encoding", total=len(LIST_IMG)):
    
    # load image
    img = load_img(str(path_img))
    # preprocessing
    img = normalization(img)
    # predict
    output = encoder(tf.expand_dims(img,axis=0))
    z_mean, z_var, z_sample = output
    z_mean = z_mean.numpy()[0] 
    z_var = z_var.numpy()[0]
    mv()

mv.to_csv(ENCODING_DIR.joinpath("encode-seq.csv"))