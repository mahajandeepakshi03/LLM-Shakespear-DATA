import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
B,T,C = 4,8,2
x=torch.randn(B,T,C)
print(x[0])
# now before giving input to the model we want the x to have the context 
# of precedding values of the x, so we do x[b,t] = mean_(i<=t) of x[b,i]

xbow=torch.zeros((B,T,C)) # zero matrix
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]
        xbow[b,t] = torch.mean(xprev,0)

print(xbow[0])

# or to optimize it we can do matrix multiplication
# take random n*n matrix of all element = 1
# then multiply that matrix with the actual one
# but then it will all the elements (not just prev one)
# for that we can choose triangular matrix 
# This part gives total, for avg do a=a/total(of a) -> as all are 1
a=torch.tril(torch.ones(3,3))
a=a/torch.sum(a,1,keepdim=True)
b=torch.randint(0,10,(3,2)).float()

# so, 'a' becomes same for all as token_size will be same 
wei=torch.tril(torch.ones(T,T))
wei=wei/wei.sum(1,keepdim=True)
x_bow2=wei @ x
print(x_bow2[0])


# another method to create wei matrix
# With softmax -> 0 will become -infinity
tril=torch.tril(torch.ones(T,T))
wei=torch.zeros((T,T))
wei=wei.masked_fill(tril==0, float('-inf')) # means whereever tril=0 put it -infinity 
wei=F.softmax(wei, dim=-1) # it works same as previous
x_bow3=wei @ x
print(x_bow3[0])


# now self attention with the context of the value at that location(token)
B,T,C = 4,8,32 #batch,time,channels
x=torch.randn(B,T,C)
head_size = 16
key=nn.Linear(C, head_size, bias=False)
query=nn.Linear(C, head_size, bias=False)
value=nn.Linear(C, head_size, bias=False)
k=key(x)        # (B,T,16)
q=query(x)      # (B,T,16)
wei=q @ k.transpose(-2,-1)     # (B,T,16) @ (B,16,T) -> (B,T,T)

tril=torch.tril(torch.ones(T,T))
# wei=torch.zeros((T,T))
wei=wei.masked_fill(tril==0, float('-inf'))
wei=F.softmax(wei, dim=-1)

v=value(x)
out=wei @ v
print(out)  # here out will have diff layers for diff tokens[i] as each token's positional encod is same but 