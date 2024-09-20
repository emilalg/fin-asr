import torch
from typing import List

class Alphabet:
    def __init__(self, alphabet_string = 'abcdefghijklmnopqrstuvwxyzåäö '):
        self.alphabet = list(alphabet_string)
        self.alphabet.append('<blank>')  # Adding blank token
        self.char_to_index = {char: index for index, char in enumerate(self.alphabet)}
        self.index_to_char = {index: char for index, char in enumerate(self.alphabet)}
        self.blank_index = len(self.alphabet) - 1  # Blank is the last index

    def text_to_array(self, text):
        return [self.char_to_index[char] for char in str.lower(text) if char in self.char_to_index]

    def array_to_text(self, array):
        return ''.join(self.index_to_char.get(int(index), '<UNK>') for index in array)

    def decode(self, log_probs: torch.Tensor, remove_blanks: bool = False) -> List[str]:
        """
        Decode log probabilities to text using simple greedy decoding.
        
        Args:
        log_probs (torch.Tensor): Log probabilities from the model
                                  Shape: (batch_size, sequence_length, num_classes)
        remove_blanks (bool): If True, remove all blank tokens from the output
        
        Returns:
        List[str]: Decoded texts for each item in the batch
        """
        # Get the most likely class at each step
        predictions = torch.argmax(log_probs, dim=-1)  # Shape: (batch_size, sequence_length)
        
        batch_texts = []
        for batch_item in predictions:
            text = self._decode_prediction(batch_item, remove_blanks)
            batch_texts.append(text)
        
        return batch_texts

    def _decode_prediction(self, prediction: torch.Tensor, remove_blanks: bool) -> str:
        """
        Decode a single prediction sequence to text.
        
        Args:
        prediction (torch.Tensor): Prediction sequence for a single item
                                   Shape: (sequence_length,)
        remove_blanks (bool): If True, remove all blank tokens from the output
        
        Returns:
        str: Decoded text
        """
        decoded = []
        previous = None
        for p in prediction:
            p = p.item()
            if remove_blanks:
                if p != self.blank_index and p != previous:
                    if p < len(self.alphabet) - 1:  # Exclude blank token
                        decoded.append(self.alphabet[p])
            else:
                if p != previous:
                    if p < len(self.alphabet):  # Include blank token
                        decoded.append(self.alphabet[p])
            previous = p
        
        return ''.join(decoded)
    