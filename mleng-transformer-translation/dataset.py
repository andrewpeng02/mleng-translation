import pickle
import random
import numpy as np

from torch.utils.data import Dataset


class ParallelLanguageDataset(Dataset):
    def __init__(self, data_path_1, data_path_2, num_tokens, max_seq_length):
        """
        Initializes the dataset
                Parameters:
                        data_path_1 (str): Path to the English pickle file processed in process-data.py
                        data_path_2 (str): Path to the French pickle file processed in process-data.py
                        num_tokens (int): Maximum number of tokens in each batch (restricted by GPU memory usage)
                        max_seq_length (int): Maximum number of tokens in each sentence pair
        """
        self.num_tokens = num_tokens
        (
            self.data_1,
            self.data_2,
            self.data_1_lengths,
            self.data_2_lengths,
            self.corresponding_data_lengths,
        ) = load_data(data_path_1, data_path_2, max_seq_length)

        self.batches = gen_batches(num_tokens, self.corresponding_data_lengths)

    def __getitem__(self, idx):
        src, src_mask = getitem(
            idx, self.data_1, self.data_1_lengths, self.batches, True
        )
        tgt, tgt_mask = getitem(
            idx, self.data_2, self.data_2_lengths, self.batches, False
        )

        return src, src_mask, tgt, tgt_mask

    def __len__(self):
        return len(self.batches)

    def shuffle_batches(self):
        self.batches = gen_batches(self.num_tokens, self.corresponding_data_lengths)


def gen_batches(num_tokens, corresponding_data_lengths):
    """
    Returns the batched data
            Parameters:
                    num_tokens (int): Maximum number of tokens in each batch (restricted by GPU memory usage)
                    data_lengths (dict): A dict with keys of tuples (length of English sentence, length of corresponding French sentence)
                                        and values of the indices that correspond to these parallel sentences
            Returns:
                    batches (List[np.array]): List of each batch (which consists of an array of indices)
    """

    # Shuffle all the indices
    for k, v in corresponding_data_lengths.items():
        random.shuffle(v)

    batches = []
    prev_tokens_in_batch = 1e10
    for k in sorted(corresponding_data_lengths):
        # v contains indices of the sentences
        v = corresponding_data_lengths[k]
        total_tokens = (k[0] + k[1]) * len(v)

        # Repeat until all the sentences in this key-value pair are in a batch
        while total_tokens > 0:
            tokens_in_batch = min(total_tokens, num_tokens) - min(
                total_tokens, num_tokens
            ) % (k[0] + k[1])
            sentences_in_batch = tokens_in_batch // (k[0] + k[1])

            # Combine with previous batch if it can fit
            if tokens_in_batch + prev_tokens_in_batch <= num_tokens:
                batches[-1].extend(v[:sentences_in_batch])
                prev_tokens_in_batch += tokens_in_batch
            else:
                batches.append(v[:sentences_in_batch])
                prev_tokens_in_batch = tokens_in_batch
            # Remove indices from v that have been added in a batch
            v = v[sentences_in_batch:]

            total_tokens = (k[0] + k[1]) * len(v)
    return batches


def pad_arr(data, max_seq_length):
    """
    Converts data into a numpy array by padding data up to max_seq_length
            Parameters:
                data (arr): Array of tokenized English sentences
                max_seq_length (int): Maximum number of tokens in each sentence pair
            Returns:
                data_arr (np.array): 2d numpy array padded with 0's
                data_lengths (np.array): 1d numpy array with the original lengths of each sub-array

    """
    data_arr = np.zeros((len(data), max_seq_length), dtype=int)
    data_lengths = np.zeros((len(data)), dtype=int)
    for i, arr in enumerate(data):
        data_arr[i][: len(arr)] = arr
        data_lengths[i] = len(arr)
    return data_arr, data_lengths


def load_data(data_path_1, data_path_2, max_seq_length):
    """
    Loads the pickle files created in preprocess-data.py
            Parameters:
                        data_path_1 (str): Path to the English pickle file processed in process-data.py
                        data_path_2 (str): Path to the French pickle file processed in process-data.py
                        max_seq_length (int): Maximum number of tokens in each sentence pair

            Returns:
                    data_1 (np.array): Array of tokenized English sentences
                    data_2 (np.array): Array of tokenized French sentences
                    data_1_lengths (np.array): Array with the original lengths of each English sentence
                    data_2_lengths (np.array): Array with the original lengths of each French sentence
                    corresponding_data_lengths (dict): A dict with keys of tuples (length of English sentence, length of corresponding French sentence)
                                         and values of the indices that correspond to these parallel sentences
    """
    with open(data_path_1, "rb") as f:
        data_1 = pickle.load(f)
    with open(data_path_2, "rb") as f:
        data_2 = pickle.load(f)

    corresponding_data_lengths = {}
    for i, (str_1, str_2) in enumerate(zip(data_1, data_2)):
        if 0 < len(str_1) <= max_seq_length and 0 < len(str_2) <= max_seq_length - 2:
            if (len(str_1), len(str_2)) in corresponding_data_lengths:
                corresponding_data_lengths[(len(str_1), len(str_2))].append(i)
            else:
                corresponding_data_lengths[(len(str_1), len(str_2))] = [i]

    data_1_arr, data_1_lengths = pad_arr(data_1, max_seq_length)
    data_2_arr, data_2_lengths = pad_arr(data_2, max_seq_length)
    return (
        data_1_arr,
        data_2_arr,
        data_1_lengths,
        data_2_lengths,
        corresponding_data_lengths,
    )


def getitem(idx, data, data_lengths, batches, src):
    """
    Retrieves a batch given an index
            Parameters:
                        idx (int): Index of the batch
                        data (np.array): Array of tokenized sentences
                        batches (List[np.array]): List of each batch (which consists of an array of indices)
                        src (bool): True if the language is the source language, False if it's the target language

            Returns:
                    batch (np.array): Array of tokenized English sentences, of size (num_sentences, num_tokens_in_sentence)
                    masks (np.array): key_padding_masks for the sentences, of size (num_sentences, num_tokens_in_sentence)
    """

    sentence_indices = batches[idx]
    batch = data[sentence_indices]
    data_lengths_batch = data_lengths[sentence_indices]
    if not src:
        # If it's in the target language, add [SOS] and [EOS] tokens
        batch = np.concatenate(
            (
                np.full((batch.shape[0], 1), 2, dtype=int),
                batch,
                np.zeros((batch.shape[0], 1), dtype=int),
            ),
            axis=1,
        )
        batch[np.arange(batch.shape[0]), data_lengths_batch + 1] = 3
        data_lengths_batch += 2
    batch = batch[:, : np.max(data_lengths_batch)]

    masks = np.full_like(batch, True, dtype=bool)
    for i in range(len(data_lengths_batch)):
        masks[i, : data_lengths_batch[i]] = False
    return batch, masks
