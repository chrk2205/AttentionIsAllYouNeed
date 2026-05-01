
from src.arguments import TrainArgs
from src.tokenize import load_train_save_bpe
from datasets import load_dataset
from src.dataset import English2HindiDataset, collate_fn
import argparse
from torch.utils.data import DataLoader
from src.trainer import LitTrainer
from src.model import Transformer
import lightning as L

# @run_and_exit(TrainArgs, description="My Training CLI")
def train(train_args : TrainArgs):

    dataset = load_dataset("Helsinki-NLP/opus-100", "en-hi", split="train")
    en_tokenizer = load_train_save_bpe(tokenizer_name="en", train_data=[v['en'] for v in dataset['translation']])
    hi_tokenizer = load_train_save_bpe(tokenizer_name="hi", train_data=[v['hi'] for v in dataset['translation']])

    train_dataset = English2HindiDataset(
        encoder_tokenizer=en_tokenizer,
        decoder_tokenizer=hi_tokenizer,
        split = "train"
    )

    validation_dataset = English2HindiDataset(
        encoder_tokenizer=en_tokenizer,
        decoder_tokenizer=hi_tokenizer,
        split="validation"
    )

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = Transformer(vocab_size=len(en_tokenizer.get_vocab()), d_model=512, d_ff=2048, num_layers=8, dropout=0.1)
    lit_model = LitTrainer(model)

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto", # Automatically detects GPU/TPU/MPS
        devices=1
    )
    trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str) 
    
    args_dict = vars(parser.parse_args())
    train_args = TrainArgs(**args_dict) # Pydantic validates here
    train(train_args)