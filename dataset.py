import torch

# import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any


class LanguagePairDataset(Dataset):
    """
    Attributes:
      dataset (Dataset): The Hugging Face dataset containing bilingual text pairs.
      tokenizer_src (Tokenizer): Tokenizer for the source language.
      tokenizer_tgt (Tokenizer): Tokenizer for the target language.
      src_lang (str): Source language code.
      tgt_lang (str): Target language code.
      seq_len (int): Sequence length for padding.
    """

    def __init__(
        self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len
    ):
        super().__init__()

        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_tkn = torch.tensor(
            [tokenizer_src.token_to_id("<sos>")], dtype=torch.int64
        )
        self.eos_tkn = torch.tensor(
            [tokenizer_src.token_to_id("<eos>")], dtype=torch.int64
        )
        self.pad_tkn = torch.tensor(
            [tokenizer_src.token_to_id("<pad>")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: Any):
        # Fetch source and target texts from the dataset
        src_txt = self.dataset[index]["translation"][self.src_lang]
        tgt_txt = self.dataset[index]["translation"][self.tgt_lang]

        # Tokenization
        encoder_input_tkns = self.tokenizer_src.encode(src_txt).ids
        decoder_input_tkns = self.tokenizer_tgt.encode(tgt_txt).ids

        # Prepare encoder and decoder inputs with special tokens
        encoder_req_pad_tkns = (
            self.seq_len - len(encoder_input_tkns) - 2
        )  # '-2' for <sos> and <eos> tokens
        enc_pad_tkn_tensors = torch.full(
            (encoder_req_pad_tkns,), self.pad_tkn.item(), dtype=torch.int64
        )

        decoder_req_pad_tkns = self.seq_len - len(decoder_input_tkns) - 1
        dec_pad_tkn_tensors = torch.full(
            (decoder_req_pad_tkns,), self.pad_tkn.item(), dtype=torch.int64
        )

        if encoder_req_pad_tkns < 0 or decoder_req_pad_tkns < 0:
            raise ValueError(
                "The sequence length after tokenization exceeds the maximum sequence length"
            )

        # Create encoder input tensor
        # Added <sos>,<eos> and <pad> tokens
        encoder_input = torch.cat(
            [
                self.sos_tkn,
                torch.tensor(encoder_input_tkns, dtype=torch.int64),
                self.eos_tkn,
                enc_pad_tkn_tensors,
            ]
        )

        # Create decoder input tensor
        # Added <sos> and <pad> tokens
        decoder_input = torch.cat(
            [
                self.sos_tkn,
                torch.tensor(decoder_input_tkns, dtype=torch.int64),
                dec_pad_tkn_tensors,
            ]
        )

        # Create the target tensor (expected output)
        # only has a <eos> token
        target = torch.cat(
            [
                torch.tensor(decoder_input_tkns, dtype=torch.int64),
                self.eos_tkn,
                dec_pad_tkn_tensors,
            ]
        )

        assert (
            encoder_input.size(0) == self.seq_len
        ), f"Encoder input length mismatch: {encoder_input.size(0)} != {self.seq_len}"
        assert (
            decoder_input.size(0) == self.seq_len
        ), f"Decoder input length mismatch: {decoder_input.size(0)} != {self.seq_len}"
        assert (
            target.size(0) == self.seq_len
        ), f"Target length mismatch: {target.size(0)} != {self.seq_len}"

        return {
            "encoder_input": encoder_input,  # (seq_len)
            # (1,1,seq_len)
            "encoder_mask": (encoder_input != self.pad_tkn)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),
            "decoder_input": decoder_input,  # (seq_len)
            # (1,1,seq_len) & (1,seq_len,seq_len)
            "causal_mask": (decoder_input != self.pad_tkn)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & generate_causal_mask(self.seq_len),
            "target": target,  # (seq_len)
            # "src_txt" : src_txt,
            # "tgt_txt" : tgt_txt
        }


def generate_causal_mask(seq_len):
    mask = torch.tril(torch.ones((1, seq_len, seq_len)), diagonal=0).type(torch.int)
    return mask == 1
