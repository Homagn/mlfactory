
#build GPT from scratch karpathy - https://www.youtube.com/watch?v=kCc8FmEb1nY

import torch
import torch.nn as nn
from torch.nn import functional as F


import os,sys
# An installation agnostic method to find and link to root of the package which is mlfactory
#==========================================================
import re
try: #testing the functions locally without pip install
  import __init__
  cimportpath = os.path.abspath(__init__.__file__)
  if 'extensions' in cimportpath:
    print("local testing ")
    import mlfactory
    cimportpath = os.path.abspath(mlfactory.__file__)

except: #testing while mlfactory is installed using pip
  print("Non local testing")
  import mlfactory
  cimportpath = os.path.abspath(mlfactory.__file__)

main_package_loc = cimportpath[:cimportpath.rfind('mlfactory')+len('mlfactory')]
print("got main package location ",main_package_loc)


os.environ['top'] = main_package_loc
sys.path.append(os.path.join(os.environ['top']))
#==========================================================


from models.pytorch.conv_reducer import Encoder as Encoder_image




# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        #print("multi head attention out shape ",out.shape)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# super simple bigram model
class visgpt(nn.Module):
    # make sure n_embd is a multiple of n_heads
    def __init__(self, n_embd = 32, block_size = 10, action_size = 9, n_heads = 8, depth = 6, dropout = 0.2, device = "cuda"): #n_embd should now be equal to the latent dimension of the variational autoencoder, block size is context length or the number of past time steps to use
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.n_embd = n_embd
        self.block_size = block_size
        self.vocab_size = action_size
        self.n_head = n_heads
        self.n_layer = depth
        self.dropout = dropout
        self.device = device
        self.enc2d = Encoder_image(shape = (256,256,1), n_hidden = n_embd)
        self.loss_type = "mse"
        
        self.enc_proj_m = nn.Sequential(nn.Linear(n_embd, n_embd),
                                    nn.Linear(n_embd, n_embd),
                                    nn.Sigmoid(), 
                                   )

        self.enc_proj_v = nn.Sequential(nn.Linear(n_embd, n_embd),
                                    nn.Linear(n_embd, n_embd),
                                    nn.Sigmoid(), 
                                   )

        assert self.n_embd%self.n_head==0, f"Please make sure n_embd is divisible by n_heads"

        #self.token_embedding_table = nn.Embedding(vocab_size, self.n_embd)
        self.laten_embedding_table = nn.Linear(self.n_embd, self.n_embd) #using linear instead of embedding this time because the latent space is already continuous

        

        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.blocks = nn.Sequential(*[Block(self.n_embd, self.n_head, self.block_size, self.dropout) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd) # final layer norm
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape[0], x.shape[1]

        # idx and targets are both (B,T) tensor of integers
        #tok_emb = self.laten_embedding_table( latent_state )#self.token_embedding_table(idx) # (B,T,C)  # Instead of embedding tokens now the input is a vector of real numbers which is the latent representation of full state
        x = x.view((B*T,1,256,256))
        enc = self.enc2d( x )

        '''
        enc_m = self.enc_proj_m(enc)
        enc_v = self.enc_proj_v(enc)
        enc_ve = torch.exp(0.5 * enc_v)

        epsilon = torch.randn_like(enc_ve)        # sampling epsilon        
        tok_emb = enc_m + enc_v*epsilon                          # reparameterization trick
        '''




        tok_emb = self.enc2d( x )
        
        tok_emb = tok_emb.view((B,T,self.n_embd))
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        #print("tok embedding shape ",tok_emb.shape)
        #print("pos embedding shape ",pos_emb.shape)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:

            if self.loss_type=="mse":
                
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                lc = logits.clone()
                tg = targets.view((-1,1))

                logits.scatter_(1, tg, 1)
                loss = F.mse_loss(lc,logits)
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                #loss_kld = - 0.5 * torch.sum(1+ enc_v - enc_m.pow(2) - enc_v.exp())
                loss = F.cross_entropy(logits, targets) #+ loss_kld

        return logits, loss



# super simple bigram model
class BigramLanguageModel(nn.Module):
    # make sure n_embd is a multiple of n_heads
    def __init__(self, n_embd = 32, block_size = 10, action_size = 9, n_heads = 8, depth = 6, dropout = 0.2, device = "cuda"): #n_embd should now be equal to the latent dimension of the variational autoencoder, block size is context length or the number of past time steps to use
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.n_embd = n_embd
        self.block_size = block_size
        self.vocab_size = action_size
        self.n_head = n_heads
        self.n_layer = depth
        self.dropout = dropout
        self.device = device

        self.loss_type = "mse"

        assert self.n_embd%self.n_head==0, f"Please make sure n_embd is divisible by n_heads"

        #self.token_embedding_table = nn.Embedding(vocab_size, self.n_embd)
        self.laten_embedding_table = nn.Linear(self.n_embd, self.n_embd) #using linear instead of embedding this time because the latent space is already continuous

        

        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.blocks = nn.Sequential(*[Block(self.n_embd, self.n_head, self.block_size, self.dropout) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd) # final layer norm
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size)

    def forward(self, latent_state, targets=None):
        B, T = latent_state.shape[0], latent_state.shape[1]

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.laten_embedding_table( latent_state )#self.token_embedding_table(idx) # (B,T,C)  # Instead of embedding tokens now the input is a vector of real numbers which is the latent representation of full state
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        #print("tok embedding shape ",tok_emb.shape)
        #print("pos embedding shape ",pos_emb.shape)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            if self.loss_type=="mse":
                '''
                B, T, C = logits.shape
                lc = logits.clone()
                #loss = torch.FloatTensor([0.0], device = self.device, requires_grad = True)
                
                #loss = torch.tensor(0.0, device = self.device)
                total_loss = torch.zeros(T, device = self.device)


                
                for t in range(T):
                    #print("shapes ",targets[:,t].shape, logits[:,t,:].shape)
                    l = logits[:,t,:]
                    tg = targets[:,t].view((-1,1))
                    
                    mse_targ = l.scatter_(1, tg, 1)
                    
                    loss = F.mse_loss(lc[:,t,:],mse_targ)
                    print("loss ",loss)
                '''

                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                lc = logits.clone()
                tg = targets.view((-1,1))

                logits.scatter_(1, tg, 1)
                loss = F.mse_loss(lc,logits)


                    
                    
                
                

            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)

        return logits, loss

    #need to rewrite because latent gpt uses different concept
    '''
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    '''

if __name__ == '__main__':


    # hyperparameters
    batch_size = 16 # how many independent sequences will we process in parallel?
    block_size = 256 # what is the maximum context length for predictions?
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    # ------------

    torch.manual_seed(1337)

    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]




    model = BigramLanguageModel()
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
