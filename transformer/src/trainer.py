import lightning as L
import torch
from torch.nn import functional as F

class LitTrainer(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]
        label = batch["label"]

        logits = self.model(encoder_input, decoder_input, encoder_mask, decoder_mask)

        # Loss Calculation
        B, T, C = logits.shape

        logits = logits.view(B*T, C)
        label = label.view(B*T)
        loss = F.cross_entropy(logits, label)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

