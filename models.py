import torch
import torch.nn as nn
import math

class VITModel(nn.Module):
    def __init__( self, vit_model, vit_model_hs, num_labels):
        super().__init__()

        self.vit_model = vit_model
        self.num_labels = num_labels
        self.classifier = nn.Linear(vit_model_hs, self.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        
    def forward(self, x, y, mask):
        outputs = self.vit_model(x, bool_masked_pos = mask)
        seq_output = outputs.sequence_output
        img_loss = outputs.loss
        logits = self.classifier(seq_output[:, 0, :])
        ce_loss = self.loss_fct(logits.view(-1, self.num_labels), y.view(-1))
        loss = img_loss + ce_loss

        return logits, loss

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class TimeSeriesModel(nn.Module):
    def __init__( self, ts_model, ts_model_hs, num_labels):
        super().__init__()

        self.ts_model = ts_model
        self.num_labels = num_labels
        self.embedding_projection = nn.Linear(1, ts_model_hs)
        self.dense = nn.Linear(ts_model_hs, ts_model_hs)
        self.classifier = nn.Linear(ts_model_hs, self.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.1)
        self.act = NewGELUActivation()
        
        
    def forward(self, x, signal_y, attention, class_y):
        x_projected = self.embedding_projection(x.unsqueeze(-1))
        outputs = self.ts_model(inputs_embeds = x_projected, attention_mask = attention, labels = signal_y, output_hidden_states = True)
        hidden_states = outputs.hidden_states
        mlm_loss = outputs.loss
        hidden_states = hidden_states[:, 0, :]
        out = self.dropout(hidden_states)
        out = self.dense(out)
        out = self.act(out)
        out = self.dropout(out)
        logits = self.classifier(out)
        ce_loss = self.loss_fct(logits.view(-1, self.num_labels), class_y.view(-1))
        loss = mlm_loss + ce_loss

        return logits, loss
    