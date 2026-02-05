import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


class English2HindiDataset(Dataset):

    def __init__(self, encoder_tokenizer, decoder_tokenizer, split = "train", seq_len = 100):

        self.dataset = load_dataset("Helsinki-NLP/opus-100", "en-hi", split=split)

        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.seq_len = seq_len


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        en_text = self.dataset[idx]['translation']['en']
        hi_text = self.dataset[idx]['translation']['hi']

        en_token_id = [ self.encoder_tokenizer.token_to_id("[SOS]") ] + self.encoder_tokenizer.encode(en_text).ids + [ self.encoder_tokenizer.token_to_id("[EOS]") ]
        label       = self.decoder_tokenizer.encode(hi_text).ids + [ self.decoder_tokenizer.token_to_id("[EOS]") ] 
        hi_token_id = [ self.decoder_tokenizer.token_to_id("[SOS]") ] + self.decoder_tokenizer.encode(hi_text).ids + [ self.decoder_tokenizer.token_to_id("[EOS]") ] 


        en_token_id_pad = en_token_id + (self.seq_len - len(en_token_id)) * [self.encoder_tokenizer.token_to_id("[PAD]")]
        hi_token_id_pad = hi_token_id + (self.seq_len - len(hi_token_id)) * [self.decoder_tokenizer.token_to_id("[PAD]")]

        en_token_id_pad = en_token_id_pad[:self.seq_len]
        hi_token_id_pad = hi_token_id_pad[:self.seq_len]

        encoder_input = torch.tensor(en_token_id_pad)
        decoder_input = torch.tensor(hi_token_id_pad)
        label         = torch.tensor(label)

        encoder_mask = ( encoder_input != torch.tensor(self.encoder_tokenizer.token_to_id("[PAD]")) ).unsqueeze(0).unsqueeze(0)
        decoder_mask = ( decoder_input != torch.tensor(self.decoder_tokenizer.token_to_id("[PAD]")) ).unsqueeze(0).unsqueeze(0)

        # casual masking
        decoder_mask = decoder_mask & causal_mask(self.seq_len).unsqueeze(0)


        return {
            "encoder_input" : encoder_input,
            "decoder_input" : decoder_input,
            "encoder_mask"  : encoder_mask,
            "decoder_mask"  : decoder_mask,
            "label"         : label
        }