

import unidecode
import string
import random
import re


all_characters = string.printable
n_characters = len(all_characters)
limit = 1000000
f = open('speeches.txt', encoding='latin-1').read()
file = unidecode.unidecode(f)
file_len = len(file)
print('file_len =', file_len)

f_s = open('shakespeare_input.txt', encoding='latin-1').read()
file_s = unidecode.unidecode(f_s)
file_len_s = len(file_s)
print('file_len_s =', file_len_s)


# To make inputs out of this big string of data, we will be splitting it into chunks.

chunk_len = 1000

def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

def random_chunk_s():
    start_index = random.randint(0, file_len_s - chunk_len)
    end_index = start_index + chunk_len + 1
    return file_s[start_index:end_index]


print(random_chunk())
print(random_chunk_s())


# # Build the Model


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size=n_characters, hidden_size=128, output_size=n_characters, n_layers=2):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        #output = F.log_softmax(output, dim=1)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))


# # Inputs and Targets


# Turn string into list of longs
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)

print(char_tensor('abcDEF'))


def random_training_set():    
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target


# # Evaluating


def category_from_output(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_characters[category_i], category_i


def evaluate(prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str
    perplexity = 1
    
    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        #char_i,top_i = category_from_output(output)
        #print(char_i,top_i)
        #Pick the one with the highest category
        
        #Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted

def perplexity(input_string):
    hidden = decoder.init_hidden()
    perplexity = 0
    n = len(input_string)
    
    inp = char_tensor(input_string[0])
    
    for p in range(n):
        output, hidden = decoder(inp, hidden)
      
        inp = char_tensor(input_string[p])
       
        output = F.softmax(output, dim=1)
        
        output_dist = output.data.view(-1).tolist()
      
        perplexity += math.log10 ( output_dist [ all_characters.index(input_string[p]) ] )
    perplexity *= -1/n

    return 10**perplexity


# # Training

# A helper to print the amount of time passed:


import time, math

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# The main training function


def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c].unsqueeze(0))

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / chunk_len


# Then we define the training parameters, instantiate the model, and start training:


n_epochs = 7000
print_every = 100
plot_every = 100
hidden_size = 100
n_layers = 2
lr = 0.005

decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0

shakespeare_perplexity = []
corpus_preplexity = []

corpus_random_chunk = random_chunk()
shakespeare_random_chunk = random_chunk_s()

#print( ' Trump Speeches Random Text: ',corpus_random_chunk)
#print( ' Shakespeare Random Text: ',shakespeare_random_chunk)
for epoch in range(1, n_epochs + 1):

    loss = train(*random_training_set())       
    loss_avg += loss

    if epoch % print_every == 0:
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
        print(evaluate('Wh', 100), '\n')
        shakespeare_perplexity.append(perplexity(random_chunk_s()))
        corpus_preplexity.append (perplexity(random_chunk()))

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0
        


#Save the PyTorch Model

models_generate = decoder
torch.save(models_generate.state_dict(), "/Users/srinivassuri/Documents/cis530/hw6/part3_submission/model_generate")


def evaluate_model(model,prime_str='A', predict_len=100, temperature=0.8):
    hidden = model.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str
    perplexity = 1
    
    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[p], hidden)
    inp = prime_input[-1]
    
    for p in range(predict_len):
        output, hidden = model(inp, hidden)
        
        #Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted

#Load the PyTorch Model back and generate few sentences
model = RNN(n_characters, hidden_size, n_characters, n_layers)
model.load_state_dict(torch.load("/Users/srinivassuri/Documents/cis530/hw6/part3_submission/model_generate"))
model.eval() #To predict

print( evaluate_model(model," ",200,0.5 ) )




#Text generated by the model
print(evaluate('Wh', 100), '\n')
print(evaluate('Wh', 100), '\n')
print(evaluate('Wh', 100), '\n')

#Code from the previous assignment for N-Grams
def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    m = len(text)
    
    #print(type(text),len(text),text)
    
    ngrams_ = []
    
    if( n == 0 ):
        for i in range(m):
            ngrams_.append( ('',text[i]) )
        return ngrams_

    for i in range(m):
        res = ''
        for j in range(i-n,i,1): #[i-n,i-1]
            if( j < 0 ):
                res += '~'
            else:
                res += text[j]
        ngrams_.append( (res,text[i]) )

        
    return ngrams_

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.vocab = set()
        self.n = n
        self.k = k
        self.counts = {}
        self.contexts ={} #Count of the Contexts
        self.contexts_tokens={} #The actual tokens that occur after the context

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        #print('Updating...',text)
        #Add the Characters to the set if they don't exist
        for ch in text:
            #print('ch is',ch)
            if ch not in self.vocab:
                self.vocab.add(ch)
                
        #Update the Counts of the ngrams
        ngrams_ = ngrams(self.n, text)

        for (context,char) in ngrams_:
            if (context,char) not in self.counts:
                self.counts[(context,char)] = 1
            else:
                self.counts[(context,char)] += 1

            if context not in self.contexts:
                self.contexts[context] = 1
                self.contexts_tokens[context] = [char]
            else:
                self.contexts[context] += 1
                if char not in self.contexts_tokens[context]:
                    words = self.contexts_tokens[context]
                    words.append(char)
                    self.contexts_tokens[context] = words
                

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        # P( char | context ) = P( context, char ) / P( context )
        if context not in self.contexts:
            #print('context not seen',context)
            return 1/(len(self.get_vocab() ))
        
        if( self.k == 0 ):
            if (context,char) not in self.counts:
                return 0.0

            return self.counts[ (context,char) ] / self.contexts[context]
        else:
            #K Smoothing
            if (context,char) not in self.counts:
                return (self.k)/(self.contexts[context]+(self.k*len(self.get_vocab()) ))

            return (self.counts[ (context,char) ] + self.k) / (self.contexts[context] + (self.k*len(self.get_vocab())) )
        

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''

        words = sorted(self.get_vocab()) #Sorted Lexicographically
        
        r = random.random()
        #print(r)
        res = 0
            
        for word in words:
            #print(word,self.prob( context,word ) )
            if res + self.prob( context,word ) > r:
                return word
            else:
                res += self.prob( context,word )

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        #print('Generate text for length',length,self.n,self.k)
        context = start_pad(self.n)
        
        text = ''
        
        for i in range(length):
            random_char = self.random_char(context)
            context = context[1:]
            context += random_char
            text += random_char
            
        return text
            

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        ngrams_ = ngrams(self.n,text)
        
        res = 0
        
        for (context,char) in ngrams_:
            prob = self.prob( context,char )
            if( prob == 0 ):
                return float('inf')
            else:
                res += math.log10( self.prob( context,char ))
        res *= -1/len(text)
        
        return 10**res

    


class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        self.vocab = set()
        self.n = n
        self.k = k
        self.counts = {}
        self.contexts ={} #Count of the Contexts
        self.contexts_tokens={} #The actual tokens that occur after the context

        self.w = []
        for i in range(self.n + 1): #[n+1] weights
             self.w.append(  1.0/(n+1) )
         

    def get_vocab(self):
        return self.vocab

    def update(self, text):
        for ch in text:
            if ch not in self.vocab:
                self.vocab.add(ch)
                
        #Update the Counts of the ngrams
        n = self.n
        while n >=0 :
            ngrams_ = ngrams(n, text)

            for (context,char) in ngrams_:
                if (context,char) not in self.counts:
                    self.counts[(context,char)] = 1
                else:
                    self.counts[(context,char)] += 1

                if context not in self.contexts:
                    self.contexts[context] = 1
                    self.contexts_tokens[context] = [char]
                else:
                    self.contexts[context] += 1
                    if char not in self.contexts_tokens[context]:
                        words = self.contexts_tokens[context]
                        words.append(char)
                        self.contexts_tokens[context] = words
            n = n-1

    def prob(self, context, char):
        res = 0
        for i in range(self.n + 1 ):
            
            if( i != 0 ):
                context = context[1:]
            
            if context not in self.contexts:
                res += ( self.w[i]*1/(len(self.get_vocab() )) )
                continue
                
            if( self.k == 0 ):
                if (context,char) not in self.counts:
                    res += 0.0
                    continue

                res +=  self.w[i] * (self.counts[ (context,char) ] / self.contexts[context] )
            else:
                #K Smoothing
                if (context,char) not in self.counts:
                    res += self.w[i] * (self.k)/(self.contexts[context]+(self.k*len(self.get_vocab()) ))
                    continue
                
                res += self.w[i] * (self.counts[ (context,char) ] + self.k) / (self.contexts[context] + (self.k*len(self.get_vocab())) )

        return res

def perplexity_model(model,input_string):
    hidden = model.init_hidden()
    perplexity = 0
    n = len(input_string)
    
    inp = char_tensor(input_string[0])
    
    for p in range(1,n,1):
        output, hidden = model(inp, hidden)
      
        inp = char_tensor(input_string[p])
       
        output = F.softmax(output, dim=1)
        
        output_dist = output.data.view(-1).tolist()
      
        perplexity += math.log10 ( output_dist [ all_characters.index(input_string[p]) ] )
    perplexity *= -1/n

    return 10**perplexity

#Create a New N-gramModel
ng = NgramModelWithInterpolation(3,2)

f1_s = open('speeches.txt', encoding='latin-1')
i = 0

for line in f1_s.readlines():
    ng.update(line)
    i +=1 


# In[322]:


print('S.no\tngram perplexity\trnn perplexity')
ng_ = []
rnn_ = []

for i in range(20):
    chunk = random_chunk()
    #print(' Chunk is: ', chunk)
    ng_.append(ng.perplexity(chunk))
    rnn_.append(perplexity_model(model,chunk ))
    print(str(i)+'\t'+str(ng_[i])+'\t'+ str(rnn_[i]) )
print('------------------------------------------------------')
print('avg:\t' ,sum(ng_)/20, '\t' , sum(rnn_)/20 )
  

