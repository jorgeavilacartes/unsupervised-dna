# https://stackoverflow.com/questions/25696615/can-i-save-a-numpy-array-as-a-16-bit-image-using-normal-enthought-python/25814423#25814423
import tensorflow as tf

class LoadImageVAE: 
    """Load Input-Output for VAE"""
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width  = img_width

    def decode_img(self, img):
        "load image as raw data, then transform it to image"
        # convert the compressed string to a 2D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=1)
        # resize the image to the desired size
        return tf.image.resize(img, [self.img_height, self.img_width])

    def load_img(self, file_path):
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img

    def __call__(self, file_path):
        "given an image path, return the input-output for the model"
        # load the raw data from the file as a string
        img = self.load_img(file_path)
        return img, img

class LoadImageEncoder(LoadImageVAE):
    
    def __call__(self, file_path,):
        "given an image path, return the input for the Encoder"
        return self.load_img(file_path)
