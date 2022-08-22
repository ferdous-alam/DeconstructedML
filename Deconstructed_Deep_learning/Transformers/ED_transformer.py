import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
import copy
from typing import Optional
from utils import TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, encoder_layer, num_encoder_layers, 
                decoder_layer, num_decoder_layers):
        super().__init__()
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_decoder_layers)

    def forward(self, src: Tensor, tgt: Tensor, 
                src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None): 

                memory = self.encoder(src, attn_mask=src_mask)
                output = self.decoder(tgt, memory, tgt_mask=tgt_mask, 
                                    memory_mask=memory_mask)

                return output


if __name__ == "__main__":
    # single encoder layer 
    enc_layer = TransformerEncoderLayer(d_model=512, num_heads=8)
    # single decoder layer
    dec_layer = TransformerDecoderLayer(d_model=512, num_heads=8)
    # encoder-decoder transformer
    transformer = Transformer(encoder_layer=enc_layer, num_encoder_layers=6, 
                                decoder_layer=dec_layer, num_decoder_layers=6)

    