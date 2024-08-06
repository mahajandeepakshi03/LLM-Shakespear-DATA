with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# using character level tokenizer (simple and easy)
chars = sorted(list(set(text)))

# encoder 
stoi = {ch:i for i, ch in enumerate(chars)}     
encode = lambda s: [stoi[c] for c in s]
# print(encode("hi theressf af"))

# decoder 
itos = {i:ch for i, ch in enumerate(chars)}     
decode = lambda l:''.join([itos[i] for i in l])
# print(decode(encode("hi theressf af")))

# Running it for the whole text file 
import torch
# Checking gpu or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = torch.tensor(encode(text), dtype=torch.long)
# print(data)

#Train-val-test split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# let's create training data (X & y) for 1st chunk
# Chunking is done here to maintain the context based knowledge 
block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    # print(f"Input : {context} and target : {target}")

# Parallel processing of multiple chunks 
torch.manual_seed(1337)
batch_size = 4      # how many chunks to maintain at one time 
block_sie = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb, yb = get_batch('train') 
# print("Inputs :")
# print(xb)
# print("Targets :")
# print(yb)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        # print(f"Input : {context} and target : {target}")

import torch.nn as nn
from torch.nn import functional as F
# Bigram Language Model 
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)    # predictions 
        
        if targets is None :    # Case of generation (not prediction)
            loss = None
        else:
            #reshape pred into 2D
            B,T,C = logits.shape                        # B : batch_size, T : chunk, C : Channel
            logits = logits.view(B*T,C)
            #reshape target into 2D
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)     # loss
        
        return logits, loss

    def generate(self, idx, max_new_tokens): 
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:,-1,:]         # (B,C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # becomes (B,1) as only first probability pf every logits will be considered
            idx = torch.concat((idx, idx_next), dim=1)      # (B, T+1)
        return idx

chars = sorted(list(set(text)))
vocab_size = len(chars)
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits, loss)
print(logits.shape)

idx=torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

# optimizer 
optimizer=torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size=32
for steps in range(100):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

# Now the deocde will be better, Also we can increase the max_token and iterations
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
