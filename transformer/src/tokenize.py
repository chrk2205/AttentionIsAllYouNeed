import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer



def load_train_save_bpe(tokenizer_name, train_data, vocab_size = 36000):
    path = f"bpe_{tokenizer_name}.json"
    
    if os.path.exists(path):
        tokenizer = Tokenizer.from_file(path)
        return tokenizer

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
    )

    tokenizer.train_from_iterator(train_data, trainer)
    tokenizer.save(path)
    return tokenizer


if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("Helsinki-NLP/opus-100", "en-hi", split="train")
    en_tokenizer = load_train_save_bpe(tokenizer_name="en", train_data=[v['en'] for v in dataset['translation']])
    hi_tokenizer = load_train_save_bpe(tokenizer_name="hi", train_data=[v['hi'] for v in dataset['translation']])