#############################################################
## ASSIGNMENT 2 CODE SKELETON
## RELEASED: 1/17/2018
## DUE: 1/24/2018
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################

from collections import defaultdict
import gzip
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from syllables import count_syllables
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    ## YOUR CODE HERE...
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for (pred,true) in zip(y_pred,y_true):
        if ( true == 1  and pred == 1 ): #True and True
            tp = tp + 1
        elif (true == 1 and  pred == 0):
            fn = fn + 1
        elif( true == 0 and pred == 0 ):
            tn = tn + 1
        elif( true == 0 and pred == 1 ):
            fp = fp + 1

    if( tp + fp == 0 ):
        return 0
    
    precision = (tp)/(tp+fp)*1.0

    return precision
    
## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    ## YOUR CODE HERE...
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for (pred,true) in zip(y_pred,y_true):
        if ( true == 1  and pred == 1 ): #True and True
            tp = tp + 1
        elif (true == 1 and  pred == 0):
            fn = fn + 1
        elif( true == 0 and pred == 0 ):
            tn = tn + 1
        elif( true == 0 and pred == 1 ):
            fp = fp + 1
            
    if( tp + fn == 0 ):
        return 0
    
    recall = (tp)/(tp+fn)*1.0

    return recall

## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    ## YOUR CODE HERE...
    recall = get_recall(y_pred, y_true)
    precision = get_precision(y_pred, y_true)

    if( recall == 0 or precision == 0 ):
        return 0
    
    fscore = (2.0 * recall * precision)/( recall + precision )
    
    return fscore

#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file):
    words = []
    labels = []   
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels

### 2.1: A very simple baseline

## Makes feature matrix for all complex
def all_complex_feature(words):
    return [ 1 ] * len(words)

## Labels every word complex
def all_complex(data_file):
    ## YOUR CODE HERE...
    words, labels = load_file(data_file)
    n = len(labels)
    y_true = labels
    y_pred = [ 1 ] * n
    
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)
    performance = [precision, recall, fscore]
    
    return performance


### 2.2: Word length thresholding


## Makes feature matrix for word_length_threshold
def length_threshold_feature(words, threshold):
    y_pred = []
    
    for word in words:
            if ( len(word) < threshold ): #Simple Word
                y_pred.append(0)
            else:
                y_pred.append(1) #Complex Word
    
    return y_pred

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    ## YOUR CODE HERE
    words, labels = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)
    threshold = 0
    threshold_limit = 18
    y_true = labels
    y_true_dev = labels_dev
    
    tperformances = []
    dperformances = []

    while ( threshold < threshold_limit ):
        y_pred = []
        y_pred_dev = []

        #Compute y_pred for training set
        for word in words:
            if ( len(word) < threshold ): #Simple Word
                y_pred.append(0)
            else:
                y_pred.append(1) #Complex Word
                
        trecall = get_recall(y_pred, y_true)
        tprecision = get_precision(y_pred, y_true)
        tfscore = get_fscore(y_pred, y_true)
        
        tperformances.append(  [tprecision, trecall,  tfscore] )

        #Compute y_pred_dev for dev set
        for word in words_dev:
            if ( len(word) < threshold ): #Simple Word
                y_pred_dev.append(0)
            else:
                y_pred_dev.append(1) #Complex Word
                
        drecall = get_recall(y_pred_dev, y_true_dev)
        dprecision = get_precision(y_pred_dev, y_true_dev)
        dfscore = get_fscore(y_pred_dev, y_true_dev)
        
        dperformances.append(  [dprecision, drecall,  dfscore] )
        threshold = threshold + 1

    #Pick the threshold with best performance
    #return tperformances, dperformances

    #Get the best weighted averaged scored of .75*training+ .25*dev\
    ans = -1
    index = -1
    for i in range(threshold_limit):
        if( (0 * tperformances[i][2]) + (1 * dperformances[i][2])  > ans ):
            ans = (0 * tperformances[i][2]) + (1 * dperformances[i][2])
            index = i

    #training_performance = [tprecision, trecall, tfscore]
    #development_performance = [dprecision, drecall, dfscore]
    #print( 'best threshold index is',index)

                
    return tperformances[index], dperformances[index]

### 2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file): 
   counts = defaultdict(int) 
   with gzip.open(ngram_counts_file, 'rt') as f: 
       for line in f:
           token, count = line.strip().split('\t') 
           if token[0].islower(): 
               counts[token] = int(count) 
   return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set

## Make feature matrix for word_frequency_threshold
def frequency_threshold_feature(words, threshold, counts):
        y_pred = []
        #Compute y_pred for training set
        for word in words:
            if counts[word] < threshold: #Complex word
                y_pred.append(1)
            else:
                y_pred.append(0) #Simple word

        return y_pred


def word_frequency_threshold(training_file, development_file, counts):
    ## YOUR CODE HERE
    words, labels = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)
    threshold = 0
    threshold_limit = 400000000
    add_shift = 50000
    y_true = labels
    y_true_dev = labels_dev
    n = 0
    
    tperformances = []
    dperformances = []

    while ( threshold <= threshold_limit ):
        
        y_pred = []
        y_pred_dev = []

        #Compute y_pred for training set
        for word in words:
            if counts[word] < threshold: #Complex word
                y_pred.append(1)
            else:
                y_pred.append(0) #Simple word
                
        trecall = get_recall(y_pred, y_true)
        tprecision = get_precision(y_pred, y_true)
        tfscore = get_fscore(y_pred, y_true)
        
        tperformances.append(  [tprecision, trecall,  tfscore] )

        #Compute y_pred_dev for dev set
        for word in words_dev:
            if counts[word] <= threshold: #Complex Word
                y_pred_dev.append(1)
            else:
                y_pred_dev.append(0) #Simple Word
                
        drecall = get_recall(y_pred_dev, y_true_dev)
        dprecision = get_precision(y_pred_dev, y_true_dev)
        dfscore = get_fscore(y_pred_dev, y_true_dev)
        
        dperformances.append(  [dprecision, drecall,  dfscore] )

        #print('training :',tperformances[threshold])
        #print('dev :',dperformances[threshold])
        #print((.5 * tperformances[n][2]) + (.5 * dperformances[n][2]))
        threshold = threshold + add_shift
        n = n + 1

    #Pick the threshold with best performance
    #return tperformances, dperformances

    #Get the best weighted averaged scored of .75*training+ .25*dev\
    ans = -1
    index = -1
    print('n is',n)
    for i in range(n):
        if( (0 * tperformances[i][2]) + (1 * dperformances[i][2])  > ans ):
            ans = (0 * tperformances[i][2]) + (1 * dperformances[i][2])
            index = i
            
    print( 'Word frequency maximum at',index * add_shift )
    #training_performance = [tprecision, trecall, tfscore]
    #development_performance = [dprecision, drecall, dfscore]
    return tperformances[index], dperformances[index]

### 2.4: Naive Bayes
def get_scaled_dev_test_training_data(words, labels,words_dev, labels_dev,counts):
    n = len(words) #Get the unique words
    X_train = np.zeros(shape=(n,2)) #no. of words x 2
    Y = labels

    n_dev = len(words_dev)
    X_dev = np.zeros(shape=(n_dev,2)) #no. of words x 2
    Y_dev = labels_dev
    
    #print('No. of words',n,X_train.shape)
    
    i=0
    for word in words:
        X_train[i][0] = len(word)
        X_train[i][1] = counts[word]
        i=i+1

    #Scale the Training Data
    std=[] #Preserve the order to scale the dev,test sets similarly
    mean=[] 
    for i in range( X_train.shape[1] ): #For each Column
        std.append( np.std(X_train[:,i]) )
        mean.append( np.mean(X_train[:,i]))
        X_train[:,i] = (X_train[:,i]-mean[i])/std[i]

    i=0
    for word in words_dev:
        X_dev[i][0] = len(word)
        X_dev[i][1] = counts[word]
        i=i+1
        
    #Scale the Dev Data
    for i in range( X_dev.shape[1] ):
        X_dev[:,i] = (X_dev[:,i]-mean[i])/std[i]
        
    return X_train,X_dev,std,mean

def get_precision_recall_fscore(y_pred,y_true):
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)
    
    return (precision,recall,fscore)

## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    ## YOUR CODE HERE
    words, labels = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)
    
    X_train, X_dev, std,mean = get_scaled_dev_test_training_data(words,
                                                                 labels,words_dev, labels_dev,counts)
    
    #Train the Classifier for the Training Set
    clf = GaussianNB()
    clf.fit(X_train, labels)
    y_pred = clf.predict(X_train)
    y_pred_dev = clf.predict(X_dev)
    
    #Calculate metrics for the train set
    training_performance = get_precision_recall_fscore(y_pred,labels)

    print('Training Performance is: ', training_performance)
    #Calculate metrics for the test set
    development_performance = get_precision_recall_fscore(y_pred_dev,labels_dev)

    return development_performance

### 2.5: Logistic Regression

## Trains a Logistic Regression classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    ## YOUR CODE HERE
    words, labels = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)
    
    X_train, X_dev, std,mean = get_scaled_dev_test_training_data(words,
                                                                 labels,words_dev, labels_dev,counts)

    #for i in range(X_train.shape[0]):
    #    print(X_train[i])
        
    #Train the Classifier for the Training Set
    clf = LogisticRegression()
    clf.fit(X_train, labels)
    y_pred = clf.predict(X_train)
    y_pred_dev = clf.predict(X_dev)

    #print('y_pred:',y_pred)
    #print('y_pred_Dev',y_pred_dev)
    #Calculate metrics for the train set
    training_performance = get_precision_recall_fscore(y_pred,labels)
    print('Training Performance is: ', training_performance)
    #print('Get the metrics for dev')
    #Calculate metrics for the test set
    development_performance = get_precision_recall_fscore(y_pred_dev,labels_dev)

    return development_performance

### 2.7: Build your own classifier

## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE

def prepare_X_y_helper(file_name, counts, testFile = False):
    #Features description:
    #1. word length
    #2. word frequency
    #3. sentence length
    #4. avg. word length in sentence
    #5. avg. word frequency
    #6. length of the biggest word in the sentence
    #7. no. of syllables in the sentence
    #8. avg. no. of syllables in the sentence
    #9. max syllable count for a word in the sentence
    #10. no. of wordnet synonyms
    #11. no. of wordnet senses
    
    with open(file_name, 'rt', encoding="utf8") as f:
        X_train = []
        labels = []
        
        i = 0
        for line in f:
            #print (line)
            #print(line[:-1])
            X = []
            if i > 0:
                line_split = line[:-1].split("\t")
                #print( line_split )
                #1.
                X.append( len(line_split[0].lower()) )
                
                #2.
                X.append( counts[line_split[0].lower()] )
                
                #3.
                if ( not testFile ):
                    sentence = line_split[3]
                else:
                    sentence = line_split[1]
                    
                X.append( len(sentence) )
                
                #4.
                
                cnt = 0
                avg_len = 0
                for word in sentence.split():
                    avg_len = avg_len + len(word)
                    cnt = cnt + 1
                    
                avg_len = avg_len/cnt * 1.0
                X.append( avg_len )
                
                #5.
                cnt = 0
                avg_len = 0
                for word in sentence.split():
                    if word in counts:
                        avg_len = avg_len + counts[word]
                        cnt = cnt + 1
                if cnt > 0 :
                    avg_len = avg_len/cnt * 1.0
                else:
                    avg_len = 0.0
                    
                X.append( avg_len )

                #6.
                len_longest_word = -1
                for word in sentence.split():
                    if len(word) > len_longest_word:
                        len_longest_word = len(word)
                    
                X.append( len_longest_word )

                #7.
                total_syllables = 0
                for word in sentence.split():
                    total_syllables = total_syllables +count_syllables( word )
                    
                X.append( total_syllables )

                #8.
                avg_syllables = 0
                cnt = 0
                for word in sentence.split():
                    avg_syllables = avg_syllables +count_syllables( word )
                    cnt = cnt + 1

                avg_syllables = avg_syllables/ cnt * 1.0
                X.append( avg_syllables )

                #9.
                max_syllable = 0
                for word in sentence.split():
                    if max_syllable < count_syllables( word ):
                        max_syllable = count_syllables( word )
                        
                X.append( max_syllable )
                
                #10.
                #11.
                X_train.append(X)
                
                if not testFile :               
                    labels.append(int(line_split[1]))
                
            i += 1
            
    if ( not testFile ):
        return X_train, labels
    else:
        return X_train
            
def prepare_X_y(training_file, development_file, counts):

    #Prepare the Training Dataset
    X_train, Y = prepare_X_y_helper(training_file, counts)
    X_train_dev, Y_dev = prepare_X_y_helper(development_file, counts)
    
    std = []
    mean = []

    #convert list of lists to numpy matrix
    X_train = np.matrix(X_train)
    Y = np.array(Y)
    X_train_dev = np.matrix(X_train_dev)
    Y_dev = np.array(Y_dev)
    
    #Scale the Training set
    for i in range( X_train.shape[1] ): #For each Column
        std.append( np.std(X_train[:,i]) )
        mean.append( np.mean(X_train[:,i]))
        X_train[:,i] = (X_train[:,i]-mean[i])/std[i]
        

    #Scale the Dev. set
    for i in range( X_train_dev.shape[1] ):
        X_train_dev[:,i] = (X_train_dev[:,i]-mean[i])/std[i]

    return X_train, Y,X_train_dev, Y_dev, std, mean
    
def classifier1(training_file, development_file, testing_file,counts):
    X_train, Y, X_dev, Y_dev, std, mean = prepare_X_y(training_file, development_file, counts)
    X_test = prepare_X_y_helper( testing_file , counts, testFile = True)

    X_test = np.matrix( X_test )
    #Scale the Test. set
    for i in range( X_test.shape[1] ):
        X_test[:,i] = (X_test[:,i]-mean[i])/std[i]

    print('Naive Bayes')
    clf = GaussianNB()
    clf.fit(X_train, Y)

    y_pred = clf.predict(X_train)


    print( get_precision_recall_fscore(y_pred,Y) )
    print( get_precision_recall_fscore(clf.predict(X_dev),Y_dev) )




    print('LogisticRegression')
    clf = LogisticRegression()
    clf.fit(X_train, Y)

    y_pred = clf.predict(X_train)


    print( get_precision_recall_fscore(y_pred,Y) )
    print( get_precision_recall_fscore(clf.predict(X_dev),Y_dev) )
    print('SVM Linear Default:')
    clf = svm.SVC()
    clf.fit(X_train, Y)

    y_pred = clf.predict(X_train)


    print( get_precision_recall_fscore(y_pred,Y) )
    print( get_precision_recall_fscore(clf.predict(X_dev),Y_dev) )

    print('SVM with Grid Search linear:')
    
    C = [0.001,0.1,1,5]
    kernels = ['linear']

    clf = GridSearchCV(svm.SVC(),cv = 10,param_grid = {"C" : C, "kernel" : kernels})
    clf.fit(X_train, Y)
    print(clf.best_params_)
    y_pred = clf.predict(X_train)

    print( get_precision_recall_fscore(y_pred,Y) )
    print( get_precision_recall_fscore(clf.predict(X_dev),Y_dev) )

    print('SVM with Grid Search rbf:')
    
    C = [0.001,0.1,1,5]
    kernels = ['rbf']

    clf = GridSearchCV(svm.SVC(),cv = 10,param_grid = {"C" : C, "kernel" : kernels})
    clf.fit(X_train, Y)
    print(clf.best_params_)
    y_pred = clf.predict(X_train)

    print( get_precision_recall_fscore(y_pred,Y) )
    print( get_precision_recall_fscore(clf.predict(X_dev),Y_dev) )

    print('Random Forests:')
    clf = RandomForestClassifier(n_estimators=50,max_depth=3)
    clf.fit(X_train, Y)
    y_pred = clf.predict(X_train)
    print( get_precision_recall_fscore(y_pred,Y) )
    print( get_precision_recall_fscore(clf.predict(X_dev),Y_dev) )

    print('Decision Trees:')
    clf = DecisionTreeClassifier()
    clf.fit(X_train, Y)
    y_pred = clf.predict(X_train)
    print( get_precision_recall_fscore(y_pred,Y) )
    print( get_precision_recall_fscore(clf.predict(X_dev),Y_dev) )

    print('AdaBoost Trees:')
    clf = GradientBoostingClassifier(n_estimators=50)
    clf.fit(X_train, Y)
    y_pred = clf.predict(X_train)
    print( get_precision_recall_fscore(y_pred,Y) )
    print( get_precision_recall_fscore(clf.predict(X_dev),Y_dev) )

    print('MLP:')
    alphas = [0.01,0.1]
    hidden_layer_sizes= [(10, ),(5,5,),(10,5,)]
    clf = GridSearchCV(MLPClassifier(max_iter=200),cv = 5,param_grid = {"alpha" : alphas, "hidden_layer_sizes" : hidden_layer_sizes})
    clf.fit(X_train, Y)
    print(clf.best_params_)
    y_pred = clf.predict(X_train)
    print( get_precision_recall_fscore(y_pred,Y) )
    print( get_precision_recall_fscore(clf.predict(X_dev),Y_dev) )
    
    #Write the output of best performer to labels.txt
    print('AdaBoost Trees:')
    clf = GradientBoostingClassifier(n_estimators=50)
    clf.fit(X_train, Y)
    y_pred = clf.predict(X_train)
    print( get_precision_recall_fscore(y_pred,Y) )
    print( get_precision_recall_fscore(clf.predict(X_dev),Y_dev) )
    y_pred = clf.predict(X_test)

    f = open('test_labels.txt', 'a')
    for y in y_pred:
        f.write(str(y)+"\n")
        
        

if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    train_data = load_file(training_file)
    
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)
