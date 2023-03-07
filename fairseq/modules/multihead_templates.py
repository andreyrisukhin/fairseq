import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch import empty

import numpy as np

# Multihead Dense Synthesizer in Einsum
class TemplatesManualMH(nn.Module):
    def __init__(self, in_dims, sentence_length, batch_size:int=1):             
        '''
        MLP to learn the weights of templates for different input tokens.
        '''
        '''
        Einsum indicies and what they represent
            b = batch size
            e = in_dims, embed dimension
            s = sentence length
            t = sentence length, distinct name to allow for 'bss' shape 'bst'
            w = hidden dimension
            n = number of templates
            h = number of heads
        
        Computation
            (1) bse (input), ew -> bsw
            (2) wn, bsw -> bsn
            (3) bsn, ns -> bss where ns: n templates, each shape s


        TODO for multihead (once above works, upgrade)
            (1) bse (input), hew -> bhsw
            (2) wn, bhsw -> bhsn
            (3) bhsn, ns -> bhss where ns: n templates, each shape s
        '''
        super(TemplatesManualMH, self).__init__() # ASK why not super().__init()__
        rng = np.random.default_rng()
        ''' Templates are fixed, not learnable. 
        New interpretation: n templates, each s, where s is set of coefficients.
            This interpretation connects to softmax attn better
            Each token associates with a set of n coefficients (token 1 -> {a[1], b[1], c[1]})
        
        # TODO Store templates as a tensor, n x s

        Big Bird templates:
        > Random attention, with 2 random elements selected per row
        > Window attention, width=3 diagonal
        > Global attention, g=2 (Rotated L to cover left and top sides)
        '''
        
        # ''' Takes length, returns torch square zero matrix with 2 elements in each row set to 1.'''
        # def template_random(n:int, r:int=2) -> torch.tensor:
        #     assert r <= n, f'Cannot have more inputs than allowed dimension'
        #     t = np.zeros((n, n))
        #     for row in t:
        #         idxs = rng.choice(n, size=2)
        #         for i in idxs:
        #             row[i] = 1
        #     return torch.tensor(t)

        # ''' Takes dim, returns a square torch matrix with a diagonal of width w. '''
        # def template_window(n:int, w:int=3):
        #     assert w <= n, f'Cannot have more inputs than allowed dimension'
        #     # From ABC repo for efficiency reasons
        #     prior_window_attn_weights = torch.zeros((n,n))
        #     radius = w//2
        #     for i in range(n):
        #         start_idx = max(0, i - radius)
        #         end_idx = min(n-1, i + radius)    
        #         length = end_idx - start_idx + 1
        #         prior_window_attn_weights[i, start_idx: end_idx + 1] = 1. / length
        #     return prior_window_attn_weights

        # ''' Takes dim, returns a square torch matrix with top/left g rows/columns assigned to 1. '''
        # def template_global(n:int, g:int=2):
        #     assert g <= n, f'Cannot have more inputs than allowed dimension'
        #     t = np.zeros((n, n))
        #     global_line = np.ones((n,))
        #     for i in range(g):
        #         t[i] = global_line.copy() # Horizontal line of 1s
        #         t.T[i] = global_line.copy() # Vertical line of 1s
        #     return torch.tensor(t)

        # t1 = template_random(n=sentence_length)
        # t2 = template_window(n=sentence_length)
        # t3 = template_global(n=sentence_length)

        ''' New templates. Subject to softmax, they each look like [s,], n of them.
        NO idea what they should be constructed as. Try (1) random, (2) front few, (3) back few.
        '''
        def v2_random(s:int):
            t = torch.rand((s,)).softmax(0)
            return t
        FRONT = 3
        BACK = 3
        def v2_first(s:int):
            t = torch.rand((s,))
            t[FRONT:] = 0 # Zero out latter
            t = torch.softmax(t,0)
            return t
        def v2_last(s:int):
            t = torch.rand((s,))
            t[:BACK] = 0 # Zero-out former
            t = torch.softmax(t,0)
            return t

        t1 = v2_random(s=sentence_length)
        t2 = v2_first(s=sentence_length)
        t3 = v2_last(s=sentence_length)

        self.templates = torch.stack((t1,t2,t3)) # [t1, t2, t3] # List of tensors 
        num_templates = len(self.templates)
        HIDDEN_DIM = in_dims

        ''' Weights and Biases for Multilayer Perceptron (linear -> relu/other activation) '''
        self.w0 = Parameter(xavier_uniform_(empty(in_dims, HIDDEN_DIM,)))  # Linear 1 weights 
        self.b0 = Parameter(constant_(empty(HIDDEN_DIM,), 0.0))  # Linear 1 bias
        self.w1 = Parameter(xavier_uniform_(empty(HIDDEN_DIM, num_templates,))) # Lin 2 weights 
        self.b1 = Parameter(constant_(empty(num_templates,), 0.0))  # Linear 2 bias; first dim instead of last, due to order of multiplication

        self.softmax = nn.Softmax(dim=-1)

        print(f'AR Template Init')
        print(f'  batch_size: {batch_size}') # ?
        print(f'  in_dims: {in_dims}') # 512
        print(f'  sentence_len: {sentence_length}') # 512  # 2048 (b/c --max-tokens) did not pass the assertion # TODO ASK HAO, unless 2048 is max input and we project down to output 512? Feels unlikely, no longer seq2seq of same length, but ask
        print(f'  hidden_dim: {HIDDEN_DIM}') # in_dims
                
        # print(f'AR DB template reasonableness: ')
        # print(f't1: {t1}')
        # print(f't2: {t2}')
        # print(f't3: {t3}')

    def forward(self, x): 
        '''
        Input the word feature vector
        Output the weights vector
        Softmax it to get linear combination (# TODO in templates_multihead_attention.py?)
            Softmax := probability distribution over attn templates
                if weights with sum to 1, output sum to 1
                Saves us having to do softmax after summing over the templates
        '''     
        '''
        The MLP is a function: output is [num_templates, ] vector weighing the fixed templates
        MLP input is [bsz, seqlen, embed_dim] 
        (all heads take the same input; different MLP for each head, init random, but take same input)
        Hidden layer is hyperparam set by dev, not tied to input or output layers
        > start with something similar to input size; questions about this are empirical, can be tuned

        MLP should work with any seq len, batch size

        Reconsider having first matrix batch size, seq len, look at synth

        MLP output is [3, ]
        '''  
            
        '''
        Parameters:
            x: Tensor (sequence length, batch size, dimension of token representation)
        Return attention, value.
        Assume that MHA.py will reuse masking machinery for template attention. Using templates_multihead_attention.py with minimal modification.
        Softmax is no longer required, because enforced with softmax on rows and on weights. TODO prove
        
        Fairseq's x [time, batch, channel] is [seq len, batch size, embed dim]. x looks like sbd
        '''      

        # MLP
        print(f'  x shape: {x.shape}')
        print(f'  w0 shape: {self.w0.shape}') 
        print(f'  b0 shape: {self.b0.shape}')
                
        hiddenReprOfTokens = torch.einsum('sbe,ew->bsw', x, self.w0) + self.b0  # x same for all heads
        filteredRepOfTokens = torch.nn.functional.relu(hiddenReprOfTokens)
        
        print(f'  fRep shape: {filteredRepOfTokens.shape}')
        print(f'  w1 shape: {self.w1.shape}') 
        print(f'  b1 shape: {self.b1.shape}')
        
        templateReprWeights = torch.einsum('wn,bsw->bsn', self.w1, filteredRepOfTokens) + self.b1
        # template_weights = self.softmax(template_weights_unbound)
        
        print(f'  templateRep shape: {templateReprWeights.shape}')

        attnWeights = torch.einsum('bsn,nt->bst', templateReprWeights, self.templates) # Just a multiplication, no parameters to learn

        # # Get Attention
        # attn_weights = (template_weights[0] * self.templates[0] + 
        #                 template_weights[1] * self.templates[1] +
        #                 template_weights[2] * self.templates[2])

        print(f'  attnWeights shape: {attnWeights.shape}')

        return attnWeights #attn_weights

        # Calculate the value using attention and x
        # value = torch.einsum('sbd,hde->bhse', x, self.value_w) + self.value_b

        # TODO modify below
        # return energy.contiguous().view((-1, self.seq_len, self.in_dims)), value.contiguous().view((-1, self.seq_len, self.head_dim))

