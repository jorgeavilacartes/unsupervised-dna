# Unsupervised Learning for DNA
Build a compress representation of DNA sequences using Variational Autoencoders and Chaos Game Representation of DNA.

- Variational Autoencoder (VAE) for Chaos Game representation for DNA
    - [x] 7-mers
    - [x] 8-mers 

- For Chaos Game Representation: https://github.com/jorgeavilacartes/complexCGR

## Implementation details: 
- VAEs Architectures are implemented using tensorflow v2.6. All architectures availables are in `unsupervised-dna/models/<name_architecture>.py`
  - current architectures are built using Convolutional (encoder) and Deconvolutional (decoder) layers. 
- To add new architectures please follow the `unsupervised-dna/models/TEMPLATE.py` code.
- Dataset to train was built with `tf.data` API [see here](https://www.tensorflow.org/guide/data)
- Data versioning with [DVC](https://dvc.org/)
- Mimic Sequences: perturbations of sequences based on [transitions and transversions](https://www.biorxiv.org/content/10.1101/2021.05.13.444008v2.full.pdf)
___ 
## In progress:
- [ ] Streamlit app to visualize results `streamlit run app.py`  
## TO DO: 
- [ ] problems with pytest: `import tensorflow as tf`
- [ ] Build a [CML](https://cml.dev/) pipeline to run experiments.
- [ ] Try with new datasets (more heterogeneous)
- [ ] Module to encapsulate dataset loading from directory. About dataset performance: ["For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required."](https://www.tensorflow.org/api_docs/python/tf/data/Dataset?hl=ko)


ESTO ES UN CAMBIO