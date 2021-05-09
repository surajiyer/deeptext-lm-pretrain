import linecache
import random
import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer


class ChatDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size
        self.data_len = 0
        self.empty_line_idx = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i, _ in enumerate(f):
                if len(_) > 0 and not _.isspace():
                    self.data_len += 1
                else:
                    self.empty_line_idx.append(i)

    def __len__(self):
        return self.data_len

    def _getitemfromfile(self, idx):
        if idx >= self.data_len:
            raise IndexError()
        idx += 1
        # TODO: check if this iteration can be optimized further?
        idx += next(i for i, j in enumerate(
            self.empty_line_idx + [float('inf')]) if j - i > idx)
        return linecache.getline(self.file_path, idx).rstrip('\n')

    def __getitem__(self, idx) -> torch.Tensor:
        text = self._getitemfromfile(idx)
        random_next_idx = -1
        while random_next_idx == -1 or random_next_idx in self.empty_line_idx:
            random_next_idx = random.randrange(self.data_len)
        idx, is_next_random = (idx + 1, 0) if random.random() > .5\
            and idx + 1 not in self.empty_line_idx\
            and idx + 1 < self.data_len\
            else (random_next_idx, 1)
        text_pair = self._getitemfromfile(idx)
        tokenized = self.tokenizer(
            text, text_pair, add_special_tokens=True, truncation=True,
            max_length=self.block_size)["input_ids"]
        return torch.tensor(tokenized, dtype=torch.long), is_next_random
