import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import util


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        pe = None
        ################################################################################
        # TODO: compute the positional encoding                                        #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe.unsqueeze(0)[:, :x.size(1)]
        return x


class HarryPotterTransformer(nn.Module):
    def __init__(self, vocab_size, feature_size, num_heads):
        super(HarryPotterTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.feature_size = feature_size
        self.num_heads = num_heads
        self.best_accuracy = -1

        self.embedding = None
        self.transformer_encoder = None
        self.decoder = None
        self.pos_encoding = None # you can omit this for Task 4

        ################################################################################
        # TODO: define the network                                                     #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.embedding = nn.Embedding(self.vocab_size,self.feature_size)
        self.pos_encoding = PositionalEncoding(self.feature_size)
        encoder_layer = nn.TransformerEncoderLayer(self.feature_size, self.num_heads, 4*self.feature_size, 0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = nn.Linear(self.feature_size,self.vocab_size)
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

    def forward(self, x):        
        attn_mask = None # you can omit this for Task 4 and Task 5
        ################################################################################
        # TODO: finish the forward pass                                                #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        attn_mask = torch.ones((x.size(1), x.size(1)), dtype=torch.float32,device=x.device)
        attn_mask.triu_(1)  
        attn_mask = torch.where(attn_mask == 1, torch.ones((x.size(1), x.size(1)),device=x.device)*float('-inf'), attn_mask) 
        
        embedded_x = self.embedding(x)
        pos_x = self.pos_encoding(embedded_x)
        pos_x = pos_x.permute(1,0,2)
        encoded_x = self.transformer_encoder(pos_x,mask = attn_mask)
        encoded_x = encoded_x.permute(1,0,2)
        x = self.decoder(encoded_x)

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        return x

    # This defines the function that gives a probability distribution and implements the temperature computation.
    def inference(self, x, temperature=1):
        x = x.view(1, -1)
        x = self.forward(x)
        x = x[0][-1].view(1, -1)
        x = x / max(temperature, 1e-20)
        x = F.softmax(x, dim=1)
        return x
    
    # Predefined loss function
    def loss(self, prediction, label, reduction='mean'):
        loss_val = F.cross_entropy(prediction.view(-1, self.vocab_size), label.view(-1), reduction=reduction)
        return loss_val

    # Saves the current model
    def save_model(self, file_path, num_to_keep=1):
        util.save(self, file_path, num_to_keep)

    # Saves the best model so far
    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if accuracy > self.best_accuracy:
            self.save_model(file_path, num_to_keep)
            self.best_accuracy = accuracy

    def load_model(self, file_path):
        util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return util.restore_latest(self, dir_path)