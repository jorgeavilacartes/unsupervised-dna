from pathlib import Path
import tensorflow as tf
from unsupervised_dna import (
    ModelLoader,
    LoadImageVAE,
)
## -- Settings -- 

# Editable
KMER = 8
VAL_SPLIT = 0.2
BATCH_SIZE = 8
EPOCHS = 30

# Default
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42
IMG_HEIGHT, IMG_WIDTH = 2**KMER, 2**KMER
DATA_DIR = Path("data/fcgr")

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
img_loader = LoadImageVAE(IMG_HEIGHT, IMG_WIDTH)

# load path to images in /data
list_ds = tf.data.Dataset.list_files(str(DATA_DIR/'*.jpg'))

# Split training and validation datasets
image_count = len(list_ds)
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(img_loader, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(img_loader, num_parallel_calls=AUTOTUNE)

# Performance of datasets
def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

# Preprocessing 
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), normalization_layer(y)))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), normalization_layer(y)))

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

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
    callbacks=[model_checkpoint_callback]
)