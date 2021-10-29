import os
import pprint
import argparse

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--wikideppaths', type=str, required=True)
parser.add_argument('--relevantdeppaths', type=str, required=True)
parser.add_argument('--outputfile', type=str, required=True)

''' 
Return a list of tuples of (hyponym, hypernym) pairs from the wiki dependence paths and identified relevantdeppaths
'''

def extractHyperHypoExtractions(wikideppaths, relevantPaths):
    '''Each line in wikideppaths contains 3 columns
        col1: word1
        col2: word2
        col3: deppath
    '''

    # Should finally contain a list of (hyponym, hypernym) tuples
    depPathExtractions = set([])

    '''
        IMPLEMENT
    '''
    all_words = {}
    with open(wikideppaths, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

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

            if (word1, word2) not in all_words:
                all_words[(word1, word2)] = {"f":0, "r":0, "n":0}


            if deppath in relevantPaths.keys():
                p =relevantPaths[deppath]
                total = sum(p.values())
                all_words[(word1, word2)]["f"] += (p["f"]/total)
                all_words[(word1, word2)]["r"] += (p["r"]/total)
                all_words[(word1, word2)]["n"] += (p["n"]/total)


    for (i,i2),v in all_words.items():
        if v["f"] == 0 and v["n"] == 0 and v["r"]==0:
            continue
        elif (0.8*v["f"]) > (v["n"]) or (0.8*v["r"]) > (v["n"]):
            if v["f"] >= v["r"]:
                depPathExtractions.add((i,i2))
            else:
                depPathExtractions.add((i2,i))

    return depPathExtractions

'''
Return the list of relevant deppath extractions reading from the file
'''
def readPaths(relevantdeppaths):
    '''
        READ THE RELEVANT DEPENDENCY PATHS HERE
    '''

    relevantPaths_dict = {'forward':set([]), 'reverse':set([])}
    with open(relevantdeppaths, 'r') as f:
        for w in f:
            w = w.strip('\n');
            path,f,r,n = w.split(" ")
            relevantPaths_dict[path] = {'f':int(f),'r':int(r),'n':int(n)}
    return relevantPaths_dict


'''
Write the hyponym hypernym pair to a file
'''
def writeHypoHyperPairsToFile(hypo_hyper_pairs, outputfile):
    directory = os.path.dirname(outputfile)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(outputfile, 'w') as f:
        for (hypo, hyper) in hypo_hyper_pairs:
            f.write(hypo + "\t" + hyper + '\n')


def main(args):
    print(args.wikideppaths)

    relevantPaths = readPaths(args.relevantdeppaths)

    hypo_hyper_pairs = extractHyperHypoExtractions(args.wikideppaths,
                                                   relevantPaths)

    writeHypoHyperPairsToFile(hypo_hyper_pairs, args.outputfile)


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
