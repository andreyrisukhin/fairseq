import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch import empty

import numpy as np


# Checkpoint loading imports below, for SVD
from fairseq import checkpoint_utils


class SVDManualMH(nn.Module):
    # Loads a model checkpoint, returns its attn matrix's svd representation for rank k.
    def import_checkpoint(self, checkpath:str, k:int=1):
        # checkpoint_utils.torch_persistent_save()
        PATH_FROM_HERE_TO_CKPT = '../wikitext/checkpoints/'
        CHECKPOINT_TO_LOAD = 'baseline_def_2/checkpoint_best.pt'
        sd = torch.load(PATH_FROM_HERE_TO_CKPT + CHECKPOINT_TO_LOAD)
    
    # https://github.com/facebookresearch/fairseq/issues/4664
    
    def __init__(self, in_dims, seq_len, heads:int=1, num_singulars:int=6): 
        ''' Parameters:
            in_dims: dimension of input representation (512)
            seq_len: length of each input instance (should be padded to be constant between samples)
            heads: the number of heads to use
            num_singulars: the number of singular values to use, = the number of rank approximation matrices to use
        '''

        ''' MLP to learn the weights of several templates for different input tokens. Templates are the rank-1 decomposition matrices of the input

        # TODO vary from sample to sample? Ask Hao about this in the meeting. I think it is okay, we are trying to learn weights that will generally be good for different samples, using the same number of top k rank approximation matrices. These should be pretty similar.
        
        Einsum indicies and what they represent
            b = batch size
            e = in_dims, embed dimension
            s = sentence length
            t = sentence length, distinct name to allow for 'bss' shape 'bst'
            w = hidden dimension
            h = number of heads
            r = number of rank-1 matrices to use (replaces n templates)
        
        Computation for multihead (different w1 and b1 per head)
            (1) bse, hew -> bhsw  Into hidden repr
            (2) wrh, bhsw -> bhsr Into template weights
            (3) bhsr, rt -> bhst   Into attention weights
            # TODO check, appears to be same as templates but with r being given by user

        Layers
            (1) Linear layer with w0 coefficients, b0 intercept. Into hidden reprsentation
                    Then apply ReLU
            (2) Linear layer with w1 coefficients, b1 intercept. Into template weights
            (3) Output attention weights, a multiplication of (constant) templates with template weights.

            # TODO check softmaxes are where they need to be

        '''
        super(SVDManualMH, self).__init__() # ASK why not super().__init()__
        SEED = 409
        rng = np.random.default_rng(SEED)
        # torch.manual_seed(SEED)
        # torch.cuda.manual_seed(SEED)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        
        # DEVICE = 'cuda'

        head_dim = in_dims // heads 
        assert (head_dim * heads == in_dims), "embed in_dims must be divisible by number of heads"
        HIDDEN_DIM = in_dims
        self.in_dims = in_dims
        self.seq_len = seq_len
        self.heads = heads
        self.num_singulars = num_singulars

        ''' Weights and Biases for Multilayer Perceptron (linear -> relu/other activation) '''
        self.w0 = Parameter(xavier_uniform_(empty(heads, in_dims, HIDDEN_DIM,)))  # Linear 1 weights 
        self.b0 = Parameter(torch.zeros(heads, HIDDEN_DIM)) # Linear 1 bias
        self.w1 = Parameter(xavier_uniform_(empty(HIDDEN_DIM, num_singulars, heads,))) # Lin 2 weights 
        self.b1 = Parameter(torch.zeros(heads, num_singulars)) # Linear 2 bias  

        self.softmax = nn.Softmax(dim=-1) # Apply softmax per row, index on "last" dimension (innermost)

        print(f'AR Template Init')
        print(f'  heads: {heads}')
        print(f'  in_dims: {in_dims}') # 512
        print(f'  seq_len: {seq_len}') # 512  # 2048 (b/c --max-tokens) did not pass the assertion # TODO ASK HAO, unless 2048 is max input and we project down to output 512? Feels unlikely, no longer seq2seq of same length, but ask
        print(f'  hidden_dim: {HIDDEN_DIM}') # in_dims

    def forward(self, x):               
        '''
        Parameters:
            x: Tensor (sequence length, batch size, dimension of token representation), the token feature vector.
            Fairseq's x [time, batch, channel] is [seq len, batch size, embed dim]. x looks like sbd
        Return attention, value.
        Assume that MHA.py will reuse masking machinery for svd attention. Using svd_multihead_attention.py with minimal modification.
        
        The MLP is a function: output is [num_templates, ] vector weighing the fixed templates
        MLP input is [bsz, seqlen, embed_dim] 
        (all heads take the same input; different MLP for each head, init random, but take same input)
        MLP should work with any seq len, batch size
        
        # TODO
        Softmax the template weights vector to get linear combination (# TODO in templates_multihead_attention.py?)
            Softmax := probability distribution over attn templates
                if weights with sum to 1, output sum to 1
                Saves us having to do softmax after summing over the templates
        '''      
                
        # Stack of rank-1 approx matrices

        # Cast x to float first

        U, S, V = torch.linalg.svd(x.float()) # pca_lowrank(x) # , q=self.num_singulars) # TODO check, is this fast enough to be worth it? # Are we instead applying SVD to an already trained regular model's weights
        s = S[:self.seq_len] #S[:self.num_singulars]
        # Does not matter which of u, v we use because dimensions n == m
        u = S[:self.seq_len] #U[:self.num_singulars]

        low_rank_x = u @ s.T
        low_rank_x = torch.broadcast_to(low_rank_x, (self.seq_len, -1, self.seq_len)) # This 1 is brittle, get feedback on flexibility here
        # low_rank_x.half()

        # TODO a good place to debug size issues 
        # assert x.size() == low_rank_x.size(), f'x has size {x.size()} while low_rank_x has size {low_rank_x.size()}'
        # Actually, this is intended behavior! We do not want to store all of x
        # Well, this may not be true.
        # Get feedback on this, for now using seq_len singular values.


        # Hang on, we need different operations I think, compared to templates

        hiddenReprOfTokens = torch.einsum('sbe,hew->bhsw', low_rank_x.half(), self.w0
                                          ) + self.b0.view(1, self.b0.size(0), 1, -1)
        filteredRepOfTokens = torch.nn.functional.relu(hiddenReprOfTokens)
        svdReprWeights = torch.einsum('wrh,bhsw->bhsr', self.w1, filteredRepOfTokens
                                           ) + self.b1.view(1, self.b1.size(0), 1, -1)
        
        softmaxedSVDReprWeights = self.softmax(svdReprWeights) # This was signaled because attn weights had negative values.

        # attnWeights = torch.einsum('bhsn,nt->bhst', softmaxedSVDReprWeights, self.svd.type_as(
        #         softmaxedSVDReprWeights)) # type depends on x type; no parameters to learn            

        return softmaxedSVDReprWeights.contiguous().view((-1, self.seq_len, self.seq_len))  #attnWeights.contiguous().view((-1, self.seq_len, self.seq_len)) 


        """
        srclen:= len of text attending to
        tgtlen:= len of text attend from
        lm, same; other applications not

        Coupling here. srclen always == seqlen in our model, b/c must pad templates

        """

        # Calculate the value using attention and x
        # value = torch.einsum('sbd,hde->bhse', x, self.value_w) + self.value_b
        # return energy.contiguous().view((-1, self.seq_len, self.in_dims)), value.contiguous().view((-1, self.seq_len, self.head_dim))

