import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


def collate_fn(batch):
    # torch.stack combines a list of tensors into a single batch tensor
    encoder_input = torch.stack([x["encoder_input"] for x in batch])
    decoder_input = torch.stack([x["decoder_input"] for x in batch])
    label         = torch.stack([x["label"] for x in batch])
    
    # Masks often have an extra dimension from unsqueeze, 
    # we want them to be [Batch, 1, Seq, Seq] or [Batch, 1, 1, Seq]
    encoder_mask  = torch.stack([x["encoder_mask"] for x in batch])
    decoder_mask  = torch.stack([x["decoder_mask"] for x in batch])

    return {
        "encoder_input": encoder_input, # [Batch, Seq_Len]
        "decoder_input": decoder_input, # [Batch, Seq_Len]
        "encoder_mask": encoder_mask,   # [Batch, 1, 1, Seq_Len]
        "decoder_mask": decoder_mask,   # [Batch, 1, Seq_Len, Seq_Len]
        "label": label                  # [Batch, Seq_Len]
    }


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
        label_pad = label + (self.seq_len - len(label)) * [self.decoder_tokenizer.token_to_id("[PAD]")]

        en_token_id_pad = en_token_id_pad[:self.seq_len]
        hi_token_id_pad = hi_token_id_pad[:self.seq_len]
        label = label_pad[:self.seq_len]

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


if __name__ == "__main__":
    from src.tokenize import load_train_save_bpe
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-hi", split="train")
    encoder_tokenizer = load_train_save_bpe(tokenizer_name="en", train_data=[v['en'] for v in dataset['translation']])
    decoder_tokenizer = load_train_save_bpe(tokenizer_name="hi", train_data=[v['hi'] for v in dataset['translation']])
    train_dataset = English2HindiDataset(encoder_tokenizer, decoder_tokenizer, split="train")
    for item in train_dataset:
        for k, v in item.items():
            print(k, v.shape)