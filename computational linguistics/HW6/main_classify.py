#main_classify.py
import codecs
import math
import random
import string
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from models import CharRNNClassify

import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

'''
    Don't change these constants for the classification task.
    You may use different copies for the sentence generation model.
'''

languages = ["af", "cn", "de", "fi", "fr", "in", "ir", "pk", "za"]
all_letters = string.ascii_letters + " .,;'"

#device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

'''
    Returns the words of the language specified by reading it from the data folder
    Returns the validation data if train is false and the train data otherwise.
    Return: A nx1 array containing the words of the specified language
'''
def getWords(baseDir, lang, train = True):
    words = []
    
    f = ''
    
    if( train ):
        #return the training data
        f = codecs.open(baseDir+lang+".txt", "r",encoding='utf-8', errors='ignore')
    else:
        #return the validation data
        f = codecs.open(baseDir+lang+"_val.txt", "r",encoding='utf-8', errors='ignore')

    lines = f.readlines()
    for line in lines:
        words.append(line.strip())

    return words

'''
    Returns a label corresponding to the language
    For example it returns an array of 0s for af
    Return: A nx1 array as integers containing index of the specified language in the "languages" array
'''
def getLabels(lang, length):
    index = languages.index(lang)
    return [index] * length

'''
    Returns all the laguages and labels after reading it from the file
    Returns the validation data if train is false and the train data otherwise.
    You may assume that the files exist in baseDir and have the same names.
    Return: X, y where X is nx1 and y is nx1
'''
def readData(baseDir, train=True):
    
    X = []
    Y = []
    
    for lang in languages:
        words = getWords(baseDir,lang, train)
        labels = getLabels(lang,len(words))
        X.extend(words)
        Y.extend(labels)
    
    return X,Y

'''
    Convert a line/word to a pytorch tensor of numbers
    Refer the tutorial in the spec
    Return: A tensor corresponding to the given line
'''
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, len(all_letters))
    n_line = len(line)
    for li in range(n_line):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

'''
    Returns the category/class of the output from the neural network
    Input: Output of the neural networks (class probabilities)
    Return: A tuple with (language, language_index)
    language: "af", "cn", etc.
    language_index: 0, 1, etc.
'''
def category_from_output(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return languages[category_i], category_i

'''
    Get a random input output pair to be used for training
    Refer the tutorial in the spec
'''
def random_training_pair(X, y):
    index = random.randint(0, len(y) - 1)
    
    category_tensor = Variable(torch.LongTensor([ y[index] ] ))
    line_tensor = Variable(line_to_tensor(X[index]))
    
    return languages[y[index]],X[index],category_tensor,line_tensor

# Just return an output given a line
def evaluate(model,line_tensor):
    hidden = model.init_hidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    
    return output

'''
    Input: trained model, a list of words, a list of class labels as integers
    Output: a list of class labels as integers
'''
def predict(model, X, y):
    labels = []
    
    for i in range(len(y)):
        
        output = evaluate(model,Variable(line_to_tensor(X[i])))
        labels.append(category_from_output(output)[1] )
    
    return labels

'''
    Input: trained model, a list of words, a list of class labels as integers
    Output: The accuracy of the given model on the given input X and target y
'''
def calculateAccuracy(model, X, y):
    
    labels = predict(model,X,y)

    matched = 0
    for i in range( len(labels)):
            if ( labels[i].item() == y[i]):
                matched += 1
                    
    return matched/len(labels)

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



'''
    Train the model for one epoch/one training word.
    Ensure that it runs within 3 seconds.
    Input: X and y are lists of words as strings and classes as integers respectively
    Returns: You may return anything
'''
def trainOneEpoch(model, criterion, optimizer, X, y):
    category, line, category_tensor, line_tensor = random_training_pair(X,y)
    
    model.zero_grad()
    hidden = model.init_hidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    
    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    #print(output,loss)
    return [output, loss.data,category, line, category_tensor, line_tensor]

'''
    Use this to train and save your classification model.
    Save your model with the filename "model_classify"
'''
def run():
    
    n_hidden = 128
    
    rnn = CharRNNClassify(len(all_letters), n_hidden, len(languages))
    
    criterion = nn.NLLLoss()
    learning_rate = 0.004
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    
    n_epochs = 450000
    print_every = 5000
    current_loss = 0
    plot_every = 1000
    all_losses = []
    
    start = time.time()
    X,y = readData('/Users/srinivassuri/Documents/cis530/hw6/train/',True)
    for epoch in range(1, n_epochs + 1):
        
        output, loss ,category, line, category_tensor, line_tensor = trainOneEpoch( rnn, criterion, optimizer, X, y)
        current_loss += loss
            
        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
                guess, guess_i = category_from_output(output)
                correct = '✓' if guess == category else '✗ (%s)' % category
                print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line, guess, correct))
                        
        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0

    #plt.figure()
    #plt.plot(all_losses)

    print('Training Accuracy:',calculateAccuracy(rnn,X,y))
    X,y = readData('/Users/srinivassuri/Documents/cis530/hw6/val/',False)
    print('Validation Accuracy:',calculateAccuracy(rnn,X,y))
    torch.save(rnn.state_dict(), '/Users/srinivassuri/Documents/cis530/hw6/part3_submission/model_classify')


