import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch import empty

import numpy as np

class TemplatesManualMH(nn.Module):
    def __init__(self, in_dims, sentence_length, heads:int=1):             
        ''' MLP to learn the weights of templates for different input tokens. '''
        '''
        Einsum indicies and what they represent
            b = batch size
            e = in_dims, embed dimension
            s = sentence length
            t = sentence length, distinct name to allow for 'bss' shape 'bst'
            w = hidden dimension
            n = number of templates
            h = number of heads
        
        Computation for multihead (different w1 and b1 per head)
            (1) bse, hew -> bhsw  Into hidden repr
            (2) wnh,bhsw->bhsn Into template weights
            (3) bhsn, nt -> bhst   Into attention weights
        '''
        super(TemplatesManualMH, self).__init__() # ASK why not super().__init()__
        SEED = 409
        rng = np.random.default_rng(SEED)
        # torch.manual_seed(SEED)
        # torch.cuda.manual_seed(SEED)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        ''' Templates are fixed, not learnable. 
        New interpretation: n templates, each s, where s is set of coefficients.
            This interpretation connects to softmax attn better
            Each token associates with a set of n coefficients (token 1 -> {a[1], b[1], c[1]})
        
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
        DEVICE = 'cuda'
        def v2_random(s:int):
            torch.manual_seed(SEED)
            t = torch.rand((s,), device=torch.device(DEVICE)).softmax(0)
            return t
        FRONT = 3
        BACK = 3
        def v2_first(s:int):
            torch.manual_seed(SEED)
            t = torch.rand((s,), device=torch.device(DEVICE))
            t[FRONT:] = 0 # Zero out latter
            t = torch.softmax(t,0)
            return t
        def v2_last(s:int):
            torch.manual_seed(SEED)
            t = torch.rand((s,), device=torch.device(DEVICE))
            t[:BACK] = 0 # Zero-out former
            t = torch.softmax(t,0)
            return t

        def template_window(s:int, w:int=3):
            assert w <= s, f'Cannot have more inputs than allowed dimension'
            # From ABC repo for efficiency reasons
            prior_window_attn_weights = torch.zeros((s,s), device=torch.device(DEVICE))
            for i in range(s):
                # TODO answer Q about padding, cold starting
                start_idx = max(0, i-w)
                prior_window_attn_weights[i, start_idx:i+1] = 1. / w 
                # TODO check this is what we want, in work.py file (off by 0x01)
            return prior_window_attn_weights

            # Old code, broke causality guarantee
            # radius = w//2
            # for i in range(s):
            #     start_idx = max(0, i - radius)
            #     end_idx = min(s-1, i + radius)    
            #     length = end_idx - start_idx + 1
            #     prior_window_attn_weights[i, start_idx: end_idx + 1] = 1. / length
            # return prior_window_attn_weights

        t1 = v2_random(s=sentence_length) # Likely want to repeat this for s times, as in big bird
        t2 = v2_first(s=sentence_length)
        t3 = v2_last(s=sentence_length)
        self.templates = torch.stack((t1,t2,t3))#.float() #.type("Float") # [t1, t2, t3] # List of tensors 
        t4 = template_window(s=sentence_length, w=3)
        self.templates = torch.cat((self.templates, t4), dim=0)

        head_dim = in_dims // heads 
        assert (head_dim * heads == in_dims), "embed in_dims must be divisible by number of heads"
        num_templates = len(self.templates)
        HIDDEN_DIM = in_dims
        self.in_dims = in_dims
        self.seq_len = sentence_length

        ''' Weights and Biases for Multilayer Perceptron (linear -> relu/other activation) '''
        self.w0 = Parameter(xavier_uniform_(empty(heads, in_dims, HIDDEN_DIM,)))  # Linear 1 weights 
        self.b0 = Parameter(constant_(empty(heads, HIDDEN_DIM,), 0.0))  # Linear 1 bias
        self.w1 = Parameter(xavier_uniform_(empty(HIDDEN_DIM, num_templates, heads,))) # Lin 2 weights 
        self.b1 = Parameter(constant_(empty(heads, num_templates,), 0.0))  # Linear 2 bias; first dim instead of last, due to order of multiplication

        self.softmax = nn.Softmax(dim=-1)

        """ March 29 2023 notes
        Templates 1) sliding window depends on pos, 2) others do not <-- these may not be sensible
        > Keep them seperate
        > 1) does not need to be indexed
        > 2) indexing. Look at batched ways to index in pytorch. High level: pass indices to tensor as another tensor, use indexing tensor
            - Easy to make mistakes
            - "Indexing a tensor using tensor/list in PyTorch"

        Once get sliding window for each location, concat with first 3 templates, then do einsum

        Template weighting may need a softmax on the weighting; put softmax on the weighting

        self.templates shape would need to be different, to have [4, <something>] with the indexed sliding window template        
        """

        print(f'AR Template Init')
        print(f'  heads: {heads}')
        print(f'  in_dims: {in_dims}') # 512
        print(f'  sentence_len: {sentence_length}') # 512  # 2048 (b/c --max-tokens) did not pass the assertion # TODO ASK HAO, unless 2048 is max input and we project down to output 512? Feels unlikely, no longer seq2seq of same length, but ask
        print(f'  hidden_dim: {HIDDEN_DIM}') # in_dims

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
        MLP should work with any seq len, batch size
        '''  
            
        '''
        Parameters:
            x: Tensor (sequence length, batch size, dimension of token representation)
            Fairseq's x [time, batch, channel] is [seq len, batch size, embed dim]. x looks like sbd
        Return attention, value.
        Assume that MHA.py will reuse masking machinery for template attention. Using templates_multihead_attention.py with minimal modification.
        Softmax is no longer required, because enforced with softmax on rows and on weights. TODO prove
        '''      
                
        hiddenReprOfTokens = torch.einsum('sbe,hew->bhsw', x, self.w0) + + self.b0.view(1, self.b0.size(0), 1, -1)
        filteredRepOfTokens = torch.nn.functional.relu(hiddenReprOfTokens)
        templateReprWeights = torch.einsum('wnh,bhsw->bhsn', self.w1, filteredRepOfTokens) + self.b1.view(1, self.b1.size(0), 1, -1)
        attnWeights = torch.einsum('bhsn,nt->bhst', templateReprWeights, self.templates.half()) # Just a multiplication, no parameters to learn

        # n x t, 1D templates <- try this for now, get baseline, if trouble
        # n x s x t (s==t), 2D templates <- big bird style
        # Could index a given row for each token; concat that vector with other templates as we are doing
        # Add a given token's window row on top of the nt templates. 
        # Could implement in a batched way! Change self.templates and 200 nt to snt, 192 use different weights to different tasks

        return attnWeights.contiguous().view((-1, self.seq_len, self.in_dims))

        # Calculate the value using attention and x
        # value = torch.einsum('sbd,hde->bhse', x, self.value_w) + self.value_b

        # TODO modify below
        # return energy.contiguous().view((-1, self.seq_len, self.in_dims)), value.contiguous().view((-1, self.seq_len, self.head_dim))

