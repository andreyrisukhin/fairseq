import torch
from torch import Tensor, nn
from torch.nn import Parameter

# AR Added
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch import empty

# Multihead Dense Synthesizer in Einsum
class SynthesizerDenseEinsumMH(nn.Module):
    def __init__(self, in_dims, sentence_length, heads:int=1):             
        
        ''' 
        Einsum indicies and what they represent
            b = batch size
            d = in_dims
            e = in_dims, distinct dimension from 'd' 
            s = sentence length
            t = sentence length, distinct dimension from 's' 
            u = sentence length, distinct values from 's', 't' (needed in second linear projection)
            h = attention heads

        TODO best style: seq_len, batch, head, ...; this is due to gains by looping through tokens when decoding
        '''
        super(SynthesizerDenseEinsumMH, self).__init__() # ASK why not super().__init()__

        # Calculate dimension per head
        head_dim = in_dims // heads 
        assert (head_dim * heads == in_dims), "embed in_dims must be divisible by number of heads"

        print(f'AR SynthDense Init')
        print(f'  head_dim: {head_dim}') # 64
        print(f'  in_dims: {in_dims}') # 512
        print(f'  sentence_len: {sentence_length}') # 512  # 2048 (b/c --max-tokens) did not pass the assertion # TODO ASK HAO, unless 2048 is max input and we project down to output 512? Feels unlikely, no longer seq2seq of same length, but ask
        print(f'  heads: {heads}') # 8

        # ''' Only used for reshaping the output '''
        self.seq_len = sentence_length
        self.in_dims = in_dims
        self.head_dim = head_dim

        ''' Weights and Biases for Linear Layers '''
        self.w0 = Parameter(xavier_uniform_(empty(heads, in_dims, sentence_length,)))  # Linear 1 weights 
        self.b0 = Parameter(constant_(empty(sentence_length,), 0.0))  # Linear 1 bias
        self.w1 = Parameter(xavier_uniform_(empty(heads, sentence_length, sentence_length,))) # Lin 2 weights 
        self.b1 = Parameter(constant_(empty(sentence_length,), 0.0))  # Linear 2 bias

        self.softmax = nn.Softmax(dim=-1)

        ''' Weights and Biases for Value Calculation '''
        self.value_w = Parameter(xavier_uniform_(empty(heads, in_dims, head_dim,))) # Used in "value" calculation
        self.value_b = Parameter(constant_(empty(head_dim,), 0.0)) # Need a bias vector for each attention head, stored here

        # print(f'Initialized SynthDenseEinsum. indims={in_dims}, seqlen={sentence_length}')

    def get_energy_dense(self, x): 
        '''
        Synthesizer Dense Energy implementation with einsum. Multihead support.
        Parameters:
            x: Tensor (sequence length, batch size, dimension of token repr)
        Output score (how likely a word corresponds to other words) matrix.
        '''       
        # print(f'DB 123')
        # print(f'Inside get_energy_dense()')     
        # print(f'  x shape: {x.shape}') 
        # print(f'  w0 shape: {self.w0.shape}')
        # print(f'  b0 shape: {self.b0.shape}')
        
        # Linear projection 1
        projectedReprOfTokens = torch.einsum('sbd,hdt->bhst', x, self.w0) + self.b0  # x same for all heads
        # changed above line when matching Fairseq inputs
        filteredRepOfTokens = torch.nn.functional.relu(projectedReprOfTokens)
        # print(f'  fRep shape: {filteredRepOfTokens.shape}')
        # print(f'  w1 shape: {self.w1.shape}') 
        # print(f'  b1 shape: {self.b1.shape}')
        
        # Linear projection 2
        energy = torch.einsum("bhst,htu->bhsu", filteredRepOfTokens, self.w1) + self.b1
        # print(f'  energy shape: {energy.shape}')
        
        return energy

    def forward(self, x): 
        '''
        Parameters:
            x: Tensor (sequence length, batch size, dimension of token representation)
        Return energy, value.
        Assume that MHA.py will reuse softmax and masking machinery for Synth and regular attention.

        Fairseq's x [time, batch, channel] is [seq len, batch size, embed dim]. x looks like sbd
        '''
        # TODO do we need to pad with zeros to match up to max-tokens? If errors, try this

        energy = self.get_energy_dense(x) 
        # attention = self.softmax(energy)  

        # print(f'value_w shape: {self.value_w.shape}')
        # print(f'value_b shape: {self.value_b.shape}')

        value = torch.einsum('sbd,hde->bhse', x, self.value_w) + self.value_b # Expect output [12/2]: (num_heads, head_dim)
        # [1/9/2023] Add bias error: size of tensor a (512) must match size of tensor b (8) at non-singleton dimension 2.

        # print(f'energy shape: {energy.shape}')
        # print(f'attention shape: {attention.shape}')
        # print(f'value shape: {value.shape}')

        # out = torch.einsum('bhsu,bhud->bhsd', attention, value) # bmm, per head (h appears in output)
        # print(f'out shape: {out.shape}')
        
        # out_combined = out.contiguous().view(-1, self.seq_len, self.in_dims)  # -1 for unknown batch size
        # print(f'out_combined shape: {out_combined.shape}')

        # return out_combined, attention
        # return energy, value 

        return energy.contiguous().view((-1, self.seq_len, self.in_dims)), value.contiguous().view((-1, self.seq_len, self.head_dim))
