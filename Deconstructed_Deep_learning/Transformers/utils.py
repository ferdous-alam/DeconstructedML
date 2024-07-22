import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
import math
import copy



class TransformerEncoder(nn.Module):
    """
    Transformer encoder is a stack of N encoder layers
    Args: 
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examnples: 
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)    
    """

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # create multiple layers 
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers


    def forward(self, src, attn_mask=None):
        output = src
        # pass the batch to loop through the number of encoder layers
        for encoder_layer in self.layers:
            output = encoder_layer(output)

        return output

    def get_attention_map(self, x, attn_mask=None):
        attention_maps = []
        for encoder_layer in self.layers:
            _, attn_map = encoder_layer.self_attn(x, x, x, 
                                        attn_mask=attn_mask, 
                                        return_attention=True)
            attention_maps.append(attn_map)

        return attn_map


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, 
                tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None):
        output = tgt
        for decoder_layer in self.layers:
            output = decoder_layer(output, memory, 
                                    tgt_mask=tgt_mask, memory_mask=memory_mask)

        return output


    def get_attention_map(self, x, memory, tgt_mask: Optional[Tensor] = None, 
                        memory_mask: Optional[Tensor] = None):
        self_attention_maps = []
        cross_attention_maps = []
        for decoder_layer in self.layers:
            _, self_attn_map = decoder_layer.self_attn(x, x, x, 
                                        attn_mask=tgt_mask, 
                                        return_attention=True)

            _, cross_attn_map = decoder_layer.cross_attn(x, memory, memory, 
                                        attn_mask=memory_mask, 
                                        return_attention=True)


            self_attention_maps.append(self_attn_map)
            cross_attention_maps.append(cross_attn_map)
            

        return self_attention_maps, cross_attention_maps


class TransformerEncoderLayer(nn.Module):
    """
    This is a single block of a transformer encoder layer, 
    this block consists of self-attention and feedforward network. 
    Here, we perform the classical encoder layer based on the 
    "Attention is all you need" paper. Also, the position of the 
    layer normalization is important in transformers implementation, 
    so here we keep two possible options for layer normalization as 
    described in the paper "On Layer Normalization in the 
    Transformer Architecture"

    args: 
        activation_fn: string, activation function name, i.e. 'relu', 'gelu' (default: 'relu')
    
    returns: 

    """

    def __init__(self, d_model, num_heads, dim_feedforward=None, 
                activation_fn='relu', 
                dropout=0.0,
                layer_norm_eps: float = 1e-5, 
                norm_first=False,
                ):
        """"
        Args: 
            d_model: 
            num_heads:
            dim_feedforward: 
            activation_fn: 
            layer_norm_eps: 
            norm_first:         
        
        returns:
            output:  

        
        """
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim=d_model, 
                                            num_heads=num_heads)

        # if ffn dimension is not given, then use 4*d_model as default value
        if dim_feedforward is None:
            dim_feedforward = 4*d_model
 
        # Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation(activation_fn)

    def _sa_block(self, x: Tensor, attn_mask):
        """
        self attention block, 
        we create Q, K, V from the input
        """
        x = self.self_attn(key=x, query=x, value=x, attn_mask=attn_mask)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor):
        """
        feed forward block
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
    def forward(self, src: Tensor, attn_mask=None, pos_encoding=False):
        x = src
        bsz, seq_len, d_model = x.size()
        
        if pos_encoding == True:
            pos_enc = PositionalEncoding(seq_len=seq_len, d_model=d_model) 
            x = pos_enc(x)

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask)
            x = x + self._ff_block(self.norm2(x))
        else: 
            x = self.norm1(x + self._sa_block(x, attn_mask))
            x = self.norm2(x + self._ff_block(x))

        return x        
            

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=None,
                activation_fn='relu', 
                dropout=0.0,
                norm_first=False,
                layer_norm_eps: float = 1e-5
                ):
        super().__init__()
        # self-attention
        self.self_attn = MultiheadAttention(embed_dim=d_model, 
                                            num_heads=num_heads, 
                                            dropout=dropout)
        # cross-attention
        self.cross_attn = MultiheadAttention(embed_dim=d_model, 
                                            num_heads=num_heads, 
                                            dropout=dropout)

        # if ffn dimension is not given, then use 4*d_model as default value
        if dim_feedforward is None:
            dim_feedforward = 4*d_model
 
        # Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)      
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation(activation_fn)

    def forward(self, tgt: Tensor, memory: Tensor, 
                tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None):
        
        x = tgt

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask)
            x = x + self._ff_block(self.norm3(x))
        else: 
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask))
            x = self.norm3(x + self._ff_block(x))

        return x        


    def _sa_block(self, x: Tensor, tgt_mask):
        """
        self attention block, 
        we create Q, K, V from the input
        """
        x = self.self_attn(query=x, key=x, value=x, attn_mask=tgt_mask)
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, memory: Tensor, 
                    memory_mask: Optional[Tensor]): 
        """
        multi-head attention block
        we create Q from the target and K, V from the memory 
        Args: 
            x: 
            memory:
            memory_mask:
        """
        x = self.cross_attn(query=x, key=memory, 
                            value=memory, attn_mask=memory_mask)

        return self.dropout2(x)

    def _ff_block(self, x: Tensor):
        """
        feed forward block
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
    




class MultiheadAttention(nn.Module):
    """
    This is the multi-head attention block
    if we have h number of heads then,
        MultiHead(Q, K, V) = Concat(head_1, head_2, ....m head_h)*W^O
    where, 
        head_i = Attention(Q*W_i^Q, K*W_i^K, V*W_i^V)  ----> single head attention
        W_i^Q, W_i^K, W_i^V ---> learnable parameters
    shape:
        query, Q: (N, L, d_k)
        key, K: (N, S, d_k)
        value, V: (N, S, d_v)

        W_i^Q: (d_model * d_k) ---> transforms input to query vectors
        W_i^K: (d_model * d_k) ---> transforms input to key vectors
        W_i^V: (d_model * d_v) ---> transforms input to value vectors

    NOTE: 
        d_k, d_v: hidden dimensions for queries/keys and values respectively
        L: target sequence length
        S: source sequence length
        d_model must be divisible by number of heads

        that ``embed_dim`` will be split 
        across ``num_heads`` (i.e. each head will have 
        dimension ``embed_dim // num_heads``).
         
    """
    def __init__(self, embed_dim, num_heads, 
                output_dim=None, kdim=None, 
                vdim=None, dropout=0.0): 
        super().__init__()
        self.embed_dim = embed_dim
        if output_dim is None:
            output_dim = self.embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "Embedding dimension must be divisible by number of heads"
        self.droptout = dropout
        # To instantiate the projection weights we need to first
        # find out whether the q, k, v matrices are equal dimensional or not
        # if they are equal:  
        # for efficient implementation stack all weight matrices 1, 2, ...., h
        # projection layer output: multiply embedding dimension by three 
        #                          because we will decouple into three 
        #                          separate Q, K, V matrices later
        # else: 
        # we need to instantiate the weight projections with proper dimensions
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        # projection matrices depend on the dimension of d_k, d_v
        if self._qkv_same_embed_dim is False: 
            # allow MHA to have different embedding dimensions when separate projection weights are used
            self.d_k = kdim // num_heads
            self.d_v = vdim // num_heads
            assert self.d_k * num_heads == self.kdim, "Embedding dimension must be divisible by number of heads"
            assert self.d_v * num_heads == self.vdim, "Embedding dimension must be divisible by number of heads"
                
            self.q_proj_weight = nn.Linear(embed_dim, self.kdim)
            self.k_proj_weight = nn.Linear(embed_dim, self.kdim)
            self.v_proj_weight = nn.Linear(embed_dim, self.vdim)
            self.output_proj_weight = nn.Linear(self.num_heads*self.d_v, 
                                            output_dim)

        else:
            self.qkv_proj_weight = nn.Linear(embed_dim, 3*embed_dim)  
            self.output_proj_weight = nn.Linear(self.num_heads*self.head_dim, 
                                                output_dim)


        # -------------------------------

        # reset parameters according to original Transformers implementation
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            # This is the original Transformer initialization
            nn.init.xavier_uniform_(self.qkv_proj_weight.weight)
            self.qkv_proj_weight.bias.data.fill_(0)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight.weight)
            nn.init.xavier_uniform_(self.k_proj_weight.weight)
            nn.init.xavier_uniform_(self.v_proj_weight.weight)
        nn.init.xavier_uniform_(self.output_proj_weight.weight)
        self.output_proj_weight.bias.data.fill_(0)


    def forward(self, query, key, value, attn_mask=None, return_attention=False):
        B, L, d_model = query.size() 
        B, S, d_model = key.size() 
        if self._qkv_same_embed_dim is False:
            q = self.q_proj_weight(query)
            k = self.k_proj_weight(key)
            v = self.v_proj_weight(value)
            q = q.reshape(B, L, self.num_heads, self.d_k)
            k = k.reshape(B, S, self.num_heads, self.d_k)
            v = v.reshape(B, S, self.num_heads, self.d_v)
            # (batch, head, seq_len, dims)
            q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

        else:
            qkv = self.qkv_proj_weight(query)
            # separate Q, K, V from linear output by reshaping 
            qkv = qkv.reshape(B, L, self.num_heads, 3*self.head_dim)
            qkv = qkv.permute(0, 2, 1, 3)   # (batch, head, seq_len, dims)
            q, k, v = qkv.chunk(3, dim=-1) # equal chunk of same size (batch, head, seq_len, embed_dim)

        # calculate values from scaled dot product attention
        # NOTE: values are calculated in parallel for multihead attention
        values, attention = _scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        values = values.permute(0, 2, 1, 3)  # (batch, seq_len, head, embed_dim)
        if self._qkv_same_embed_dim is False:
            values = values.reshape(B, L, self.num_heads * self.d_v)
        else: 
            values = values.reshape(B, L, self.embed_dim)
        output = self.output_proj_weight(values)
        if return_attention:
            return output, attention
        else:
            return output


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model): 
        super().__init__()
        # create matrix of (seq_length, hidden_dim) representing the positional encoding 
        pos_enc = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denominatior = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * denominatior)
        pos_enc[:, 1::2] = torch.cos(position * denominatior)
        pos_enc = pos_enc.unsqueeze(0)

        self.register_buffer('pos_enc', pos_enc, persistent=False)

    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1)]
        return x



def _get_clones(module, N):
    layers = nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    return layers


def _get_activation(activation_fn):
    if activation_fn == "relu":
        return F.relu 
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError('activation must be relu/gelu')


def _scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor]=None):
    """
    B: batch size, 
    h: number of heads, 
    l_z: target sequence length, 
    l_x: target sequence length, 
    d_k: embedding dimension of key/query
    d_v: embedding dimension of value

    shape: 
        q: (B, h, l_x, d_k) 
        k: (B, h, l_z, d_k) 
        v: (B, h, l_z, d_v)
        attn_mask: (l_x, l_z), this is a tensor of 1s or zeros based on where we want to mask
        output: (B, h, l_x, d_v) 

    Args: 
        q: query matrix
        k: key matrix 
        v: value matrix, 
        attn_mask: attn mask matrix

    output: 
        values: attention values after multiplying value matrix with softmax output
        attn: attention values after softmax

    NOTE: 
    attn_mask is given as the mask we would want for a single sequence
    This mask MUST be repeated along the batch and heads dimension
    As we are using 4-dimensional tensors as input, we no longer can use 
    the badbmm of pytorch to calculate the attention, rather 
    we use the equivalent of badbmm as a combination of matmul and add
    """

    B, h, d_k = q.size()[0], q.size()[1], q.size()[-1]

    # calculate attention logits 
    attn_logits = torch.matmul(q, k.transpose(-2, -1))

    # scaling 
    attn_logits = attn_logits / math.sqrt(d_k)

    if attn_mask is not None:
        # mask all false elements in the attention mask with -inf values 
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))
        attn_mask = attn_mask.repeat(B, h, 1, 1)  # repeat mask to match the attention logtis tensor shape
        attn_logits = torch.add(attn_mask, attn_logits)

    # softmax of attention logits
    attn = F.softmax(attn_logits, dim=-1)
    # (B, Nt, Ns) x (B, Bs, d_k) --> (B, Nt, d_v)
    values = torch.matmul(attn, v)
    return values, attn

