from pathlib import Path
import tensorflow as tf

from unsupervised_dna import (
    LoadImageEncoder, 
    LoadImageVAE,
)

AUTOTUNE = tf.data.AUTOTUNE

class DatasetVAE:    

    def __init__(self, data_dir: Path, batch_size: int, kmer: int, shuffle: bool = True):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.kmer = kmer
        self.img_loader = self.get_img_loader()
        self.charge_dataset()

    def charge_dataset(self,):
        # load path to images in /data
        self.ds = tf.data.Dataset.list_files(str(self.data_dir/'*.jpg'))
        n_files = len(self.ds)
        print("dataset loaded from: {} | Total files: {}".format(str(self.data_dir), n_files))

    def split_train_val(self, val_size):
        # Split training and validation datasets
        image_count = len(self.ds)
        val_size = int(image_count * val_size)
        train_ds = self.ds.skip(val_size)
        val_ds = self.ds.take(val_size)

        return train_ds, val_ds

    def charge_img_loader(self, ds):
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        ds = ds.map(self.img_loader, num_parallel_calls=AUTOTUNE)
        return ds
    
    def configure_for_performance(self, ds):
        # Performance of datasets
        ds = ds.cache()
        if self.shuffle is True: 
            ds = ds.shuffle(buffer_size=len(ds))
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def preprocessing(self, ds):
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        ds = ds.map(lambda x, y: (normalization_layer(x), normalization_layer(y)))
        return ds

    def get_img_loader(self,):
        return LoadImageVAE(2**self.kmer, 2**self.kmer)
        

    def __call__(self, for_training: bool=True, val_size: float=0.2):
        """
        given a directory, returns train and val sets if 'for_training == True'
        otherwise, returns one dataset
        """

        if for_training is True:
            train_ds, val_ds = self.split_train_val(val_size)
            
            # img loader
            train_ds = self.charge_img_loader(train_ds)
            val_ds = self.charge_img_loader(val_ds)

            # performance
            train_ds = self.configure_for_performance(train_ds)
            val_ds = self.configure_for_performance(val_ds)

            # preprocessing
            train_ds = self.preprocessing(train_ds)
            val_ds = self.preprocessing(val_ds)

            return train_ds, val_ds            

        else:
            # img loader
            ds = self.charge_img_loader(self.ds)
            # performance
            ds = self.configure_for_performance(ds)
            # preprocessing
            ds = self.preprocessing(ds)
            return ds


class DatasetEncoder(DatasetVAE):

    def __init__(self, data_dir: Path, batch_size: int, kmer: int, shuffle: bool = False):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.kmer = kmer
        self.img_loader = self.get_img_loader()
        self.charge_dataset()

    def __call__(self,):
        # img loader
        ds = self.charge_img_loader(self.ds)
        # performance
        ds = self.configure_for_performance(ds)
        # preprocessing
        ds = self.preprocessing(ds)
        return ds

    def get_img_loader(self,):
        return LoadImageEncoder(2**self.kmer, 2**self.kmer)

    def preprocessing(self, ds):
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        ds = ds.map(lambda x: normalization_layer(x))
        return ds
