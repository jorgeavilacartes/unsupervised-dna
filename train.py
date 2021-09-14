from tensorflow.python.ops.gen_batch_ops import batch
from parameters import PARAMETERS

from pathlib import Path
import tensorflow as tf
from unsupervised_dna import (
    ModelLoader,
    LoadImageVAE,
    DatasetVAE,
)
## -- Settings -- 

# Editable
KMER = PARAMETERS["KMER"]
VAL_SPLIT = PARAMETERS["VAL_SPLIT"]
BATCH_SIZE = PARAMETERS["BATCH_SIZE"]
EPOCHS = PARAMETERS["EPOCHS"]

# Default
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42
IMG_HEIGHT, IMG_WIDTH = 2**KMER, 2**KMER
DATA_DIR = Path(f"data/fcgr-{KMER}-mer")

## -- Distributed training -- 
# try:
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver().connect()
#     strategy = tf.distribute.experimental.TPUStrategy(tpu)

# except ValueError: # detect one or multoĺe GPUs
#     strategy = tf.distribute.MirroredStrategy()

## -- Model Selection -- 
loader = ModelLoader()
model = loader("vae_{}mer".format(KMER))

## -- Datasets -- 
ds_vae = DatasetVAE(DATA_DIR, batch_size=BATCH_SIZE, kmer=KMER, shuffle=True)
train_ds, val_ds = ds_vae(for_training=True, val_size=0.2)

## -- Callbacks --
#checkpoint_filepath = Path('checkpoint/{epoch:02d}-{val_loss:.2f}.hdf5')
checkpoint_path = "checkpoint/cp.ckpt"
#checkpoint_filepath.mkdir(parents=True,exist_ok=True)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, #'./training/checkpoint/{epoch:02d}-{val_loss:.2f}.hdf5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        model_checkpoint_callback,
        tf.keras.callbacks.EarlyStopping(patience=5),
        ]
)