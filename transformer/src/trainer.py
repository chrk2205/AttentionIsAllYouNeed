import lightning as L
import torch
from torch.nn import functional as F

class LitTrainer(L.LightningModule):
    def __init__(self, model, tokenizer_en, tokenizer_hi):
        super().__init__()
        self.model = model
        self.tokenizer_en = tokenizer_en
        self.tokenizer_hi = tokenizer_hi

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

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]
        original_label = batch["label"]

        logits = self.model(encoder_input, decoder_input, encoder_mask, decoder_mask)

        # Loss Calculation
        B, T, C = logits.shape

        logits = logits.view(B*T, C)
        label = original_label.view(B*T)
        loss = F.cross_entropy(logits, label)
        
        self.log("valid_loss", loss, prog_bar=True)

        start_token = self.tokenizer_hi.token_to_id("[SOS]")
        end_token = self.tokenizer_hi.token_to_id("[EOS]")

        inference_results = self.model.generate(encoder_input, encoder_mask, start_token, end_token)
        for inference_result, ground_truth in zip(inference_results, original_label):
            gt = self.tokenizer_hi.decode(ground_truth.tolist())
            ifr = self.tokenizer_hi.decode(inference_result.tolist())

            print(f"inference result {ifr}")
            print(f"Ground truth result {gt}")

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, eps=1e-9)
        
        # Linear warmup followed by cosine decay
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-4,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1, # Warmup for first 10% of training
            anneal_strategy='cos'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "interval": "step",
                "scheduler": scheduler
            }
        }
