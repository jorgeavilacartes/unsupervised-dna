import tensorflow as tf

def kl_divergence(z_mean, z_log_var):
    kl_loss = -0.5*(1+z_log_var-tf.square(z_mean) - tf.exp(z_log_var))
    return tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

def reconstruction_loss(real, reconstruction): 
    return tf.reduce_mean(tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(real, reconstruction), axis=(1,2))
    )