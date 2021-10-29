from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support

import pickle
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import nltk
from conlleval import evaluate
from conlleval import EvalCounts
from conlleval import parse_args
from conlleval import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import SparsePCA
# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


def getfeats(word, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    
    #f1
    if word.isupper():
        up_word = '1'
    else:
        up_word = '0'

    #f2
    if word[0].isupper():
        cap = '1'
    else:
        cap = '0'

    word_shape = ''
    for letter in word:
        if( letter.isdigit() ):
            word_shape += 'd'
        elif(letter.islower() ):
            word_shape += 'x'
        elif(letter.isupper() ):
            word_shape += 'X'
        else:
            word_shape += letter

    if word.islower():
        lower = '1'
    else:
        lower = '0'

    if '-' in word:
        hyphen = '1'
    else:
        hyphen = '0'

    if  '\''  in word:
        aps = '1'
    else:
        aps = '0'

    features = [
        (o + 'word', word),
        (o+ 'upper',up_word),
        (o + 'cap',cap),
        (o + 'ws',word_shape),
        (o + 'lower',lower),
        (o + 'hyphen',hyphen),
        (o + 'aps',aps),
        (o + 'wl', len(word) )
        #(o+'pt',nltk.pos_tag([word])[0][1] )
    ]
    return features
    

def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    for o in range(-5,5,1):
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            featlist = getfeats(word, o)
            features.extend(featlist)
    
    return dict(features)

if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))
    
    train_feats = []
    train_labels = []

    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    # TODO: play with other models
    #model = Perceptron(verbose=1)
    #model = DecisionTreeClassifier()
    #model = RandomForestClassifier(n_estimators = 25)
    #model = LogisticRegression()
    #model = LinearSVC()
    #model.fit(X_train, train_labels)

    #models = [MultinomialNB(),LinearSVC(),Perceptron(alpha=0.1),RandomForestClassifier(n_estimators=20),RandomForestClassifier(),DecisionTreeClassifier(),LogisticRegression(),LogisticRegression(C=0.01)]

    #names = ['mnb','svc','percpa0p01','rf_20','rf','dt','logreg','logreg_lr_0c01']
    models = [LinearSVC()]
    names = ['svc_pca']
    accuracies=[]
    fscores=[]
    precisions=[]
    recalls=[]

    preds = ["train_","dev_","test_"]
    datas = [train_sents,dev_sents,test_sents]
    data_eval = []
    names_eval = []

    for model,name in zip(models,names):
        print('Fitting model',name)
                  
        model.fit(X_train, train_labels)
        for pred_,data in zip(preds,datas):
            test_feats = []
            test_labels = []
            
            # switch to test_sents for your final results
            for sent in data:
                for i in range(len(sent)):
                    feats = word2features(sent,i)
                    test_feats.append(feats)
                    test_labels.append(sent[i][-1])

            X_test = vectorizer.transform(test_feats)
         
            y_pred = model.predict(X_test)

            j = 0
            #print("Writing to "+ pred_+"results.txt")
            # format is: word gold pred
            with open(pred_+"results.txt", "w") as out:
                for sent in data:
                    for i in range(len(sent)):
                        word = sent[i][0]
                        gold = sent[i][-1]
                        pred = y_pred[j]
                        j += 1
                        out.write("{}\t{}\t{}\n".format(word,gold,pred))
                out.write("\n")
        
            if pred_ == "test_":
                continue

            #Evaluate the model
            print('Evaluating the ' + pred_ + ' set')
            arg = [pred_+"results.txt"]
            args = parse_args(arg)


            c = evaluate(open(args.file), args)
            overall, by_type = metrics(c)
            accuracies.append( c.correct_tags/c.token_counter)
            precisions.append(100.*overall.prec)
            recalls.append(100.*overall.rec)
            fscores.append(100.*overall.fscore)
            data_eval.append( pred_.replace("_", ""))
            names_eval.append(name)

            print('Accuracy',c.correct_tags/c.token_counter)
            print('Precision',100.*overall.prec)
            print('Recall',100.*overall.rec)
            print('f-score',100.*overall.fscore)

    print('Model\tData-Type\tAccuracy\tPrecision\tRecall\tFScore\t')
    n = len(models)
    for i1 in range(2*n):
        print(names_eval[i1] +"\t"+data_eval[i1]+"\t"+ str(format(accuracies[i1], '.2f'))+"\t"+str(format(precisions[i1], '.2f'))+"\t"+str(format(recalls[i1], '.2f'))+"\t"+str(format(fscores[i1], '.2f'))   )
        #print(names[i1%n]+"\t"+data_eval[i1]+"\t"+str(accuracies[i1])+"\t"+str(precisions[i1])+"\t"+str(recalls[i1])+"\t"+str(fscores[i1])+"\n")

    #print('Saving Classifier')
    #pickle.dump(model, open('model', 'wb'))








