from src.dataset import SerialTextDataset
from src.model import BigramLanguageModel
from src.trainer import LitTrainer
from torch.utils.data import DataLoader
import lightning as L
import torch


def train(
    batch_size = 32,
    max_steps = 10000,
    block_size = 32
):

    with open('dataset.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    vocab = sorted(list(set(text)))

    # tokenizers
    stoi = { ch:i for i,ch in enumerate(vocab) }
    itos = { i:ch for i,ch in enumerate(vocab) }


    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: [itos[i] for i in l]


    train_text = text[:int(0.9*len(text))]
    valid_text = text[int(0.9*len(text))+1:]



    train_dataset = SerialTextDataset(train_text, block_size, encode)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    valid_dataset = SerialTextDataset(valid_text, block_size, encode)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)


    model = BigramLanguageModel(vocab_size=len(vocab), block_size = block_size)
    lit_model = LitTrainer(model)

    trainer = L.Trainer(
        max_steps=max_steps,
        accelerator="auto", # Automatically detects GPU/TPU/MPS
        devices=1
    )
    trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    print(''.join(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=2000)[0].tolist())))

    trainer.save_checkpoint("model.pt")



if __name__ == "__main__":
    train()