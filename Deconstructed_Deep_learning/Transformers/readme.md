## Documentation 

This is a modular implementation of the Transformer architecture. It can be thought of a simplified version of the pytorch transformer implementation. Only certain places have been modified. 

Explanation of notations: 
1. $d_\text{model}:$ embedding dimension of each input element of the sequence
2. $\text{num\_heads}:$ number of attention heads to implement multi-heads
3. $\text{enc\_layer}:$ number of encoder layers to be implemented
4. $\text{dec\_layer}:$ number of decoder layers to be implemented

First, define a custom encoder layer. The **utils.py** file contains a straighforward implementation of the original transformer encoder layer. You can use that as the following. 

        from utils import TransformerEncoderLayer 
        # single encoder layer 
        enc_layer = TransformerEncoderLayer(d_model=512, num_heads=8)

Similarly, we can define a decoder layer as well. 

        from utils import TransformerDecoderLayer 
        # single decoder layer
        dec_layer = TransformerDecoderLayer(d_model=512, num_heads=8)

## Encoder-Decoder Transformer

Using the above building blocks we can define our own transformer class. GPT or BERT like models can be built using their respective architectures. For example, encoder-only architecture for BERT, decoder-only architecture for GPT. Note that in the original BERT or GPT implementation the encoder and decoder layers have slightly different architecture than the original transformer papers. So, we will need to modify the $\text{TransformerEncoderLayer}$ and $\text{TransformerDecoderLayer}$ accordingly.  

For the original transformer implementation, I already written the encoder-decoder transformer class. 

        transformer = Transformer(encoder_layer=enc_layer, num_encoder_layers=6, 
                                    decoder_layer=dec_layer, num_decoder_layers=6)

Examples, 

        X = torch.rand(32, 70, 512)
        mask = torch.triu(torch.ones(70, 70)).transpose(0, 1)
        Y = transformer(X, X, src_mask=mask, tgt_mask=mask, memory_mask=mask)