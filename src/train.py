
from src.arguments import TrainArgs
from src.tokenize import load_train_save_bpe
from datasets import load_dataset
from src.dataset import English2HindiDataset
import argparse

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
    

    for item in validation_dataset:
        pass

    for item in train_dataset:
        pass

    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str) 
    
    args_dict = vars(parser.parse_args())
    train_args = TrainArgs(**args_dict) # Pydantic validates here
    train(train_args)