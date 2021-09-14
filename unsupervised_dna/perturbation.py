# Perturbation of DNA sequences adapted from
# https://github.com/millanp95/DeLUCS/blob/master/src/mimics.py
import numpy as np

class MimicSequence: 
    "Given a sequence, perturbate it applying transitions or/and transversions"
    def __init__(self, p_transition: float = 1-10**-4, p_transversion: float = 1-0.5*10**-4):
        self.p_transition = p_transition # Probability of NO transition.
        self.p_transversion = p_transversion # Probability of NO Transversion.

    def __call__(self, seq, transition: bool = True, transversion: bool = True):
        
        # Find mutations 
        if transition and transversion:
            index, mutations = self.transition_transversion(seq)
        elif transition and not transversion:
            index, mutations = self.transition(seq)
        elif not transition and transversion:
            index, mutations = self.transversion(seq)
        else: 
            # If transitions and transversions are set to False, return the original sequence
            return seq
        
        # apply mutations to the sequence
        for j,mutation in zip(index, mutations):
            seq = self.replacer(seq, mutation, j)

        return seq

    def transition(self, seq,):
        """Mutate Genomic sequence using transitions only.

        Args:
            seq (str): Original Genomic Sequence.

        Returns:
            [tuple]: mutations and positions (index)
        """        
        x = np.random.random(len(seq))
        index = np.where(x > self.p_transition)[0]
        mutations = []

        for i in index:
            nb = seq[i]
            if nb == 'A':
                mutations.append('G')
            elif nb == 'G':
                mutations.append('A')
            elif nb == 'T':
                mutations.append('C')
            elif nb == 'C':
                mutations.append('T')
            else: 
                mutations.append(nb)
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
            nb = seq[i]
            random_number = np.random.uniform()
            if nb == 'A':
                mutation = 'T' if random_number > 0.5 else 'C'
                    
            elif nb == 'G':
                mutation = 'T' if random_number > 0.5 else 'C'
                
            elif nb == 'T':
                mutation = 'A' if random_number > 0.5 else 'G'
            
            elif nb == 'C':
                mutation = 'A' if random_number > 0.5 else 'G'
            else: 
                mutation = nb

            mutations.append(mutation)

        return index, mutations


    def transition_transversion(self, seq,):
        """Mutate Genomic sequence using transitions and transversions

        Args:
            seq (str): sequence 

        Returns:
            str: new sequence
        """        
        # First transitions.
        idx, mutations = self.transition(seq,)
        seq = list(seq)
        # Then transversions.
        for (i, new_bp) in zip(idx, mutations):
            seq[i] = new_bp

        return self.transversion(''.join(seq),)

    @staticmethod
    def replacer(seq, mutation, index,):
        """insert a mutation in position 'index' of sequence

        Args:
            s (str): sequence
            mutation (str): character to be replaced
            index ([type]): position in sequence where to replace the mutation

        Raises:
            ValueError: when index is out of bounds

        Returns:
            str: new sequence with mutations
        """        
        # raise an error if index is outside of the string
        if index not in range(len(seq)):
            raise ValueError("index outside given string")

        # insert the new string between "slices" of the original
        return seq[:index] + mutation + seq[index + 1:]