from torch.utils.data import Dataset
import torch



class SerialTextDataset(Dataset):
    def __init__(self, text, block_size, encoder) -> None:
        super().__init__()
        self.text = text
        self.block_size = block_size
        self.encoder = encoder

    def __len__(self):
        return int(len(self.text)-self.block_size)

    def __getitem__(self, index) :
        x = torch.tensor(self.encoder(self.text[index:index+self.block_size]))
        y = torch.tensor(self.encoder(self.text[index+1:index+self.block_size+1]))
        return x, y


if __name__ == "__main__":
    
    with open('dataset.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"length of dataset {len(text)}")


    vocab = sorted(list(set(text)))

    # tokenizers
    stoi = { ch:i for i,ch in enumerate(vocab) }
    itos = { i:ch for i,ch in enumerate(vocab) }

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: [itos[i] for i in l]


    dataset = SerialTextDataset(text, 8, encode)
    for i in dataset:
        # print(i)
        continue