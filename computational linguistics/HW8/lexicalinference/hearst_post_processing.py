import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

porter = PorterStemmer()
file_r = open("../hearst.txt","r")
file_r_lines = file_r.readlines()

#Hearst Pattern1
#Using nltk. Just output each word as a separate hypernym/hyponym
file_p1 =  open("../hearst_p.txt","w")


#Hearst Pattern1
#Using nltk. Output only the Nouns in the second column as relation pairs


# Post Processing
i = 0
for line in file_r_lines:
    words = line.split('\t')
    main_word = (words[0].split(' ')[-1])
    sub_words = (words[1].split(' '))

    '''
    text = nltk.word_tokenize(words[1].replace('\n',''))
    pos_tags = nltk.pos_tag(text)
    for  (word,tag) in pos_tags:
        if tag == "NN":
            file_p1.write( main_word)+ '\t' + word + '\n');
    '''

    for word in sub_words[-1:]:

            file_p1.write( main_word + '\t' + word.replace('\n','') + '\n');


file_p1.close()
