from unsupervised_dna import __version__
from unsupervised_dna.models.vae_kmer import get_model

def test_version():
    assert __version__ == '0.1.0'

def test_vae():
    model = get_model()
    print(model.summary())