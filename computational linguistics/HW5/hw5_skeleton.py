import math, random

################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

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
            

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

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

    
################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

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
################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################


    
class NgramModelMyModel(NgramModelWithInterpolation): 
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k, code):
        self.vocab = set()
        self.n = n
        self.k = k
        self.counts = {}
        self.contexts ={} #Count of the Contexts
        self.contexts_tokens={} #The actual tokens that occur after the context
        self.code = code

        self.w = []
        for i in range(self.n + 1): #[n+1] weights
             self.w.append(  1.0/(n+1) )
             
    def set_w(self,index,val):
        self.w[index] = val

    def update_all(self):
        f = open('./train/' + self.code + '.txt','rb')
        lines = f.readlines()
        
        for line in lines:
            try:
                self.update(line.decode('utf-8').strip())
            except UnicodeDecodeError:
                continue

#Check for Accuracy for each country
n = len(COUNTRY_CODES)

def compute(code,file_path,ngrams,k_smoothing):
    models = []
    ans = []
    accuracy = 0
    total = 0
    
    for i in range(n):
        models.append( NgramModelMyModel(n=ngrams,k=k_smoothing,code=COUNTRY_CODES[i]) )
        models[i].update_all()
        ans.append(0)

    f = open(file_path,'rb')
    lines = f.readlines()

    for line in lines:
        res = float('inf')
        index = 0
        total += 1
        for i in range(n):
            
            try:
                if( not isinstance(line,str) ):
                    line =   line.decode('utf-8')
                
                if( models[i].perplexity(line.strip()) < res ):
                    res = models[i].perplexity(line.strip())
                    index = i
            
            except UnicodeDecodeError:
                continue
            
        ans[index] += 1
        
    #print( COUNTRY_CODES[code] + '\t' +str(ans[code]/total) )
    return ans[code]/total

def compute_test(file_path,ngrams,k_smoothing):
    models = []
    accuracy = 0
    total = 0
    
    for i in range(n):
        models.append( NgramModelMyModel(n=ngrams,k=k_smoothing,code=COUNTRY_CODES[i]) )
        models[i].update_all()

    f = open(file_path,'rb')
    lines = f.readlines()

    f1 = open('test_labels.txt','w')
    for line in lines:
        res = float('inf')
        index = 0
        total += 1
        for i in range(n):
            
            try:
                if( not isinstance(line,str) ):
                    line =   line.decode('utf-8')
                
                if( models[i].perplexity(line.strip()) < res ):
                    res = models[i].perplexity(line.strip())
                    index = i
            
            except UnicodeDecodeError:
                continue
        f1.write( COUNTRY_CODES[index] +'\n')
    f1.close()


#Actual Test-File
'''
for k1 in range(5):
    for k2 in range(5):

        avg = 0
        print('City\tAccuracy\tn,k:',k1,k2)
        for i in range(n):
            avg += compute(i,'./val/'+COUNTRY_CODES[i]+'.txt',k1,k2)
        avg /=n
        print('Over all Accuracy:',avg)


print('based on the best found values')
compute_test('cities_test.txt',4,1)
'''

if __name__ == '__main__':
    pass
