import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch import empty

import numpy as np

class TemplatesManualMH(nn.Module):
    def __init__(self, in_dims, sentence_length, heads:int=1):             
        ''' MLP to learn the weights of templates for different input tokens. 
        
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

        Layers
            (1) Linear layer with w0 coefficients, b0 intercept. Into hidden reprsentation
                    Then apply ReLU
            (2) Linear layer with w1 coefficients, b1 intercept. Into template weights
            (3) Output attention weights, a multiplication of (constant) templates with template weights.

            # TODO check softmaxes are where they need to be

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
        
        ''' Takes length, returns torch square zero matrix with 2 elements in each row set to 1.'''
        def template_random(n:int, r:int=2) -> torch.tensor:
            assert r <= n, f'Cannot have more inputs than allowed dimension'
            t = np.zeros((n, n))
            for i, row in enumerate(t):
                idxs = rng.choice(n, size=r) # made this causal below. THINK how forcing selecting n changes distribution with this condition.
                idxs = [num for num in idxs if num <= i] # Causal mask imposed, only keep those idxs less than or on diagonal to avoid lookahead. We keep original distribution and mask it. This <= diag condition allows self attention, which is still causual.
                # print(f'DB row {i} idx {idxs}')
                count_valid = len(idxs)
                for idx in idxs:
                    row[idx] = 1 / count_valid # Normalize
            return torch.tensor(t, device=torch.device(DEVICE))

        ''' Takes dim, returns a square torch matrix with top/left g rows/columns assigned to 1. '''
        def template_global(n:int, g:int=2): # TODO violates causality
            assert g <= n, f'Cannot have more inputs than allowed dimension'
            t = np.zeros((n, n))
            global_line = np.ones((n,))
            for i in range(g):
                t[i] = global_line.copy() # Horizontal line of 1s
                t.T[i] = global_line.copy() # Vertical line of 1s

            # Causal mask for diagonal (slow, could be improved)
            for i, row in enumerate(t):
                row[i+1:] = 0
                row /= sum(row) # Normalize rows
            return torch.tensor(t, device=torch.device(DEVICE))

        DEVICE = 'cuda'
        # FRONT = 3
        # def v2_first(s:int):
        #     torch.manual_seed(SEED)
        #     t = torch.rand((s,), device=torch.device(DEVICE))
        #     t[FRONT:] = 0 # Zero out latter # TODO breaks causality for first FRONT tokens
        #     t = torch.softmax(t,0)
        #     return t

        def template_window(s:int, w:int=3):
            assert w <= s, f'Cannot have more inputs than allowed dimension'
            prior_window_attn_weights = torch.zeros((s,s), device=torch.device(DEVICE))
            for i in range(s):
                # TODO padding, cold starting?
                start_idx = max(0, (i-w)+1)
                prior_window_attn_weights[i, start_idx:i+1] = 1. / (i+1 - start_idx)
            return prior_window_attn_weights

        # t2 = v2_first(s=sentence_length)
        t4 = template_window(s=sentence_length, w=3)
        t5 = template_window(s=sentence_length, w=50)

        t_r1 = template_random(sentence_length, r=1)
        t_r2 = template_random(sentence_length, r=2)
        t_r50 = template_random(sentence_length, r=50)

        t_g1 = template_global(sentence_length, g=1)
        t_g3 = template_global(sentence_length, g=3)

        t_w3 = template_window(sentence_length, w=3)
        t_w50 = template_window(sentence_length, w=50)

        # t5 = templ

        # print(f't_r50 {t_r50}')
        # print(f't_g3 {t_g3}')

        # self.templates = torch.cat((t4, torch.broadcast_to(t2, (len(t2), len(t2)))), dim=0)
        # self.templates = t4
        # self.templates = torch.cat((t4, t5), dim=0)
        self.templates = torch.cat((
            t_r1, t_r2, t_r50,
            t_g1, t_g3,
            t_w3, t_w50
        ), dim=0)

        head_dim = in_dims // heads 
        assert (head_dim * heads == in_dims), "embed in_dims must be divisible by number of heads"
        num_templates = len(self.templates)
        print(f'num_templates={num_templates}, true 2d templates represented={num_templates / sentence_length}')
        HIDDEN_DIM = in_dims
        self.in_dims = in_dims
        self.seq_len = sentence_length

        ''' Weights and Biases for Multilayer Perceptron (linear -> relu/other activation) '''
        self.w0 = Parameter(xavier_uniform_(empty(heads, in_dims, HIDDEN_DIM,)))  # Linear 1 weights 
        self.b0 = Parameter(torch.zeros(heads, HIDDEN_DIM)) #(constant_(empty(heads, HIDDEN_DIM,), 0.0))  # Linear 1 bias
        self.w1 = Parameter(xavier_uniform_(empty(HIDDEN_DIM, num_templates, heads,))) # Lin 2 weights 
        self.b1 = Parameter(torch.zeros(heads, num_templates))  #(constant_(empty(heads, num_templates,), 0.0))  # Linear 2 bias; first dim instead of last, due to order of multiplication

        self.softmax = nn.Softmax(dim=-1)

        print(f'AR Template Init')
        print(f'  heads: {heads}')
        print(f'  in_dims: {in_dims}') # 512
        print(f'  sentence_len: {sentence_length}') # 512  # 2048 (b/c --max-tokens) did not pass the assertion # TODO ASK HAO, unless 2048 is max input and we project down to output 512? Feels unlikely, no longer seq2seq of same length, but ask
        print(f'  hidden_dim: {HIDDEN_DIM}') # in_dims

    def forward(self, x):               
        '''
        Parameters:
            x: Tensor (sequence length, batch size, dimension of token representation), the token feature vector.
            Fairseq's x [time, batch, channel] is [seq len, batch size, embed dim]. x looks like sbd
        Return attention, value.
        Assume that MHA.py will reuse masking machinery for template attention. Using templates_multihead_attention.py with minimal modification.
        Softmax is no longer required, because enforced with softmax on rows and on weights. TODO prove

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
                
        hiddenReprOfTokens = torch.einsum('sbe,hew->bhsw', x, self.w0
                                          ) + self.b0.view(1, self.b0.size(0), 1, -1)
        filteredRepOfTokens = torch.nn.functional.relu(hiddenReprOfTokens)
        templateReprWeights = torch.einsum('wnh,bhsw->bhsn', self.w1, filteredRepOfTokens
                                           ) + self.b1.view(1, self.b1.size(0), 1, -1)
        
        # for 1 head, 1 bsz, the template weights are (s,n). 
            # For third token, MLP(3) -> [z1^[3], z2^[3], ... (if more templates)]
            # Each column is the weights for each token of a template
            # If two templates, sliding window (s,s) and first token (s,1), the third token's template is: z1^[3] * Sw[3] + z2^[3] * ft
            # Extending to all tokens, multiply all of z1 @ Sw + z2 @ ft, broadcast ft
        # Impose softmax constraints: (1) sum(s column) == 1, (2) each s column item > 0
        # Then normalize at end after template multiplication

        # Concat templates and multiply by z output (likely easier, just make sure I document)
        attnWeights = torch.einsum('bhsn,nt->bhst', templateReprWeights, self.templates.type_as(
                templateReprWeights)) # type depends on x type; no parameters to learn

        return attnWeights.contiguous().view((-1, self.seq_len, self.in_dims))

        # Calculate the value using attention and x
        # value = torch.einsum('sbd,hde->bhse', x, self.value_w) + self.value_b
        # return energy.contiguous().view((-1, self.seq_len, self.in_dims)), value.contiguous().view((-1, self.seq_len, self.head_dim))

