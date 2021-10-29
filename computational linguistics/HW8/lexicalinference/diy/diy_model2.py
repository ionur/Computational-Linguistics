from pymagnitude import *
vectors = Magnitude("GoogleNews-vectors-negative300.magnitude")




#Query the distance, Classify as hypernym-hyponym if they are less than a distance.


threshold = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
counts = [0 for i in range(10)]
i=0

for i in range(10):
    print('Threshold >= ',i)
    file_p1 = open('./bless2011/diy_'+str(i)+'.txt',"w")

    #gather vocab from trainining wikipedia_sentences
    path = './bless2011/data_lex_test.tsv'
    f = open(path,"r")
    lines = f.readlines()

    for line in lines:
        words = line.split('\t')
        sim = vectors.similarity(words[0].replace('\n',''), words[1].replace('\n',''))

        if( sim >= threshold[i] ):
            #print( words[0].replace('\n',''), words[1].replace('\n',''), sim)
            counts[i] = counts[i] + 1
            file_p1.write( words[0].replace('\n','') + '\t' + words[1].replace('\n','') + '\n')

    f.close()

    file_p1.close()

print( counts )
