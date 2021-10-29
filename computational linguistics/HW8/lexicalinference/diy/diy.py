from pymagnitude import *
vectors = Magnitude("GoogleNews-vectors-negative300.magnitude")

print( len(vectors) )

print( vectors.most_similar("alligator", topn = 15)) # Most similar by key )
