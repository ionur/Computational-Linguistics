import os
import pprint
import argparse
import collections

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()
parser.add_argument('--wikideppaths', type=str, required=True)
parser.add_argument('--trfile', type=str, required=True)

parser.add_argument('--outputfile', type=str, required=True)

'''
Use word pair labels to extract relevant deppaths and write to file based on a dictionary count
'''
def extractRelevantPaths(wikideppaths, wordpairs_labels, outputfile):
    '''Each line in wikideppaths contains 3 columns
        col1: word1
        col2: word2
        col3: deppath
    '''

    lines_read = 0

    relevantPathsCount = {}

    with open(wikideppaths, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            lines_read += 1

            word1, word2, deppath = line.split("\t")

            '''
                IMPLEMENT METHOD TO EXTRACT RELEVANT DEPEDENCY PATHS HERE

                Make sure to be clear about X being a hypernym/hyponym.

                Dependency Paths can be extracted in multiple different categories, such as
                1. Forward Paths: X is hyponym, Y is hypernym
                2. Reverse Paths: X is hypernym, Y is hyponym
                3. Negative Paths: If this path exists, definitely not a hyper/hyponym relations
                4. etc......
            '''

            if (word1, word2) in wordpairs_labels.keys() and (word2, word1) in wordpairs_labels.keys():
                if deppath not in relevantPathsCount.keys():
                    relevantPathsCount[deppath] = {'forward':0,'reverse':0,'negative':0}

                if wordpairs_labels[(word1, word2)] == 'False' and wordpairs_labels[(word2, word1)] == 'False':
                    relevantPathsCount[deppath]['negative'] += 1
                elif wordpairs_labels[(word1, word2)] == 'True' and wordpairs_labels[(word2, word1)] == 'False':
                    relevantPathsCount[deppath]['forward'] += 1
                elif wordpairs_labels[(word1, word2)] == 'False' and wordpairs_labels[(word2, word1)] == 'True':
                    relevantPathsCount[deppath]['reverse'] += 1
            elif (word1, word2) in wordpairs_labels.keys():
                if wordpairs_labels[(word1, word2)] == 'True':
                    if deppath not in relevantPathsCount.keys():
                        relevantPathsCount[deppath] = {'forward': 0, 'reverse': 0, 'negative': 0}
                    relevantPathsCount[deppath]['forward'] += 1


            elif (word2, word1) in wordpairs_labels.keys():
                if wordpairs_labels[(word2, word1)] == 'True':
                    if deppath not in relevantPathsCount.keys():
                        relevantPathsCount[deppath] = {'forward': 0, 'reverse': 0, 'negative': 0}
                    relevantPathsCount[deppath]['reverse'] += 1



    with open(outputfile, 'w') as f:
        for deppath,v in relevantPathsCount.items():
            f.write(deppath+" "+str(v['forward'])+" "+str(v['reverse'])+" "+str(v['negative']) )
            f.write('\n')

    return relevantPathsCount.keys()


def readVocab(vocabfile):
    vocab = set()
    with open(vocabfile, 'r') as f:
        for w in f:
            if w.strip() == '':
                continue
            vocab.add(w.strip())
    return vocab


def readWordPairsLabels(datafile):
    wordpairs = {}
    with open(datafile, 'r') as f:
        inputdata = f.read().strip()

    inputdata = inputdata.split("\n")
    for line in inputdata:
        word1, word2, label = line.strip().split('\t')
        word1 = word1.strip()
        word2 = word2.strip()
        wordpairs[(word1, word2)] = label
    return wordpairs


def main(args):
    print(args.wikideppaths)

    wordpairs_labels = readWordPairsLabels(args.trfile)

    #only get the word pairs that are hypernym, hyponym
    extractRelevantPaths(args.wikideppaths, wordpairs_labels, args.outputfile)


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
