# Perturbation of DNA sequences from
# https://github.com/millanp95/DeLUCS/blob/master/src/mimics.py
import numpy as np

class MimicSequence: 
    "Given a sequence, perturbate it applying transitions or/and transversions"
    def __init__(self, p_transition: float = 1-10**-4, p_transversion: float = 1-0.5*10**-4):
        self.p_transition = p_transition # Probability of NO transition.
        self.p_transversion = p_transversion # Probability of NO Transversion.

    def transition(self, seq,):
        """
        Mutate Genomic sequence using transitions only.
        :param seq: Original Genomic Sequence.
        :param threshold: probability of NO Transition.
        :return: Mutated Sequence.
        """
        x = np.random.random(len(seq))
        index = np.where(x > self.p_transition)[0]
        mutations = []

        for i in index:
            nucleotide = seq[i]
            if nucleotide == 'A':
                mutations.append('G')
            if nucleotide == 'G':
                mutations.append('A')
            if nucleotide == 'T':
                mutations.append('C')
            if nucleotide == 'C':
                mutations.append('T')

        return index, mutations

    def transversion(self, seq: str):
        """Mutate Genomic sequence using transversions only.

        Args:
            seq (str): Original Genomic Sequence

        Returns:
            str: Mutated sequence
        """        
        x = np.random.random(len(seq))
        index = np.where(x > self.p_transversion)[0]
        mutations = []

        for i in index:
            nucleotide = seq[i]

            if nucleotide == 'A':
                random_number = np.random.uniform()
                if random_number > 0.5:
                    mutations.append('T')
                else:
                    mutations.append('C')
            if nucleotide == 'G':
                random_number = np.random.uniform()
                if random_number > 0.5:
                    mutations.append('T')
                else:
                    mutations.append('C')
            if nucleotide == 'T':
                random_number = np.random.uniform()
                if random_number > 0.5:
                    mutations.append('A')
                else:
                    mutations.append('G')
            if nucleotide == 'C':
                random_number = np.random.uniform()
                if random_number > 0.5:
                    mutations.append('A')
                else:
                    mutations.append('G')

        return index, mutations


    def transition_transversion(self, seq, threshold_1, threshold_2):
        """
        Mutate Genomic sequence using transitions and transversions
        :param seq: Original Sequence.
        :param threshold_1: 
        :param threshold_2: Probability of NO transversion.
        :return:
        """
        # First transitions.
        idx, mutations = self.transition(seq, threshold_1)
        seq = list(seq)
        # Then transversions.
        for (i, new_bp) in zip(idx, mutations):
            seq[i] = new_bp

        return self.transversion(''.join(seq), threshold_2)