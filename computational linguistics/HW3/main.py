import os
import csv
import subprocess
import re
import random
import numpy as np


def read_in_shakespeare():
  '''Reads in the Shakespeare dataset processesit into a list of tuples.
     Also reads in the vocab and play name lists from files.
  Each tuple consists of
  tuple[0]: The name of the play
  tuple[1] A line from the play as a list of tokenized words.
  Returns:
    tuples: A list of tuples in the above format.
    document_names: A list of the plays present in the corpus.
    vocab: A list of all tokens in the vocabulary.
  '''

  tuples = []

  with open('will_play_text.csv') as f:
    csv_reader = csv.reader(f, delimiter=';')
    for row in csv_reader:
      play_name = row[1]
      line = row[5]
      line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
      line_tokens = [token.lower() for token in line_tokens]

      tuples.append((play_name, line_tokens))

  with open('vocab.txt') as f:
    vocab =  [line.strip() for line in f]

  with open('play_names.txt') as f:
    document_names =  [line.strip() for line in f]

  return tuples, document_names, vocab

def get_row_vector(matrix, row_id):
  return matrix[row_id, :]

def get_column_vector(matrix, col_id):
  return matrix[:, col_id]

def create_term_document_matrix(line_tuples, document_names, vocab):
  '''Returns a numpy array containing the term document matrix for the input lines.
  Inputs:
    line_tuples: A list of tuples, containing the name of the document and 
    a tokenized line from that document.
    document_names: A list of the document names
    vocab: A list of the tokens in the vocabulary
  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.
  Let m = len(vocab) and n = len(document_names).
  Returns:
    td_matrix: A mxn numpy array where the number of rows is the number of words
        and each column corresponds to a document. A_ij contains the
        frequency with which word i occurs in document j.
  '''

  vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
  docname_to_id = dict(zip(document_names, range(0, len(document_names))))

  # YOUR CODE HERE
  m = len(vocab)
  n = len(document_names)
  td_matrix = np.zeros(shape=(m,n))

  for (doc_name,words) in line_tuples:
    for word in words:
      #print(word,vocab_to_id[word],docname_to_id[word])
      td_matrix[ vocab_to_id[word] , docname_to_id[doc_name] ] = td_matrix[ vocab_to_id[word] , docname_to_id[doc_name] ] + 1

  return td_matrix

def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
  '''Returns a numpy array containing the term context matrix for the input lines.
  Inputs:
    line_tuples: A list of tuples, containing the name of the document and 
    a tokenized line from that document.
    vocab: A list of the tokens in the vocabulary
  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.
  Let n = len(vocab).
  Returns:
    tc_matrix: A nxn numpy array where A_ij contains the frequency with which
        word j was found within context_window_size to the left or right of
        word i in any sentence in the tuples.
  '''

  vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
  
  # YOUR CODE HERE
  n = len(vocab)
  #print( 'total number of words',n)
  tc_matrix = np.zeros(shape=(n,n))
  for (doc_name,words) in line_tuples:
    m = len(words)
    
    for i in range(m):
      for j in range(i-context_window_size,i+context_window_size+1,1):
        #from [i-window_size] to [i+window_size]

        if( i == j or j < 0 or j >= m):
          continue
        else:
          tc_matrix[ vocab_to_id[words[i]],vocab_to_id[words[j]] ] = tc_matrix[ vocab_to_id[words[i]],vocab_to_id[words[j]] ] + 1

      
  return tc_matrix

def create_PPMI_matrix(term_context_matrix):
  '''Given a term context matrix, output a PPMI matrix.
  
  See section 15.1 in the textbook.
  
  Hint: Use numpy matrix and vector operations to speed up implementation.
  
  Input:
    term_context_matrix: A nxn numpy array, where n is
        the numer of tokens in the vocab.
  
  Returns: A nxn numpy matrix, where A_ij is equal to the
     point-wise mutual information between the ith word
     and the jth word in the term_context_matrix.
  '''
  # YOUR CODE HERE
  term_context_matrix += 1 
  n = term_context_matrix.shape[0]
  sum_fij = np.sum( term_context_matrix )
  #print( 'Sum is: ', sum_fij )

  #Compute pi*
  pi_ = [] #pi*
  for i in range(n):
    pi_.append( np.sum( term_context_matrix[i,:] )/ sum_fij  )
    
  #Compute p*j
  p_j = [] 
  for i in range(n):
    p_j.append( np.sum( term_context_matrix[:,i] )/ sum_fij )
    
  ppmi = term_context_matrix[:]

  #Added the Smoothing
  ppmi /= sum_fij
  
  for i in range(n):
    if( pi_[i] == 0 ):
      ppmi[i,:] *= 0
    else:
      ppmi[i,:] /= pi_[i]
  
  for i in range(n):
    if( p_j[i] == 0 ):
      ppmi[:,i] *= 0
    else:
      ppmi[:,i] /= p_j[i]

  ppmi = np.log2( ppmi )

  ppmi = ppmi.clip(min=0)
  
  print( 'Finished Computing the values' )
  return ppmi

def create_tf_idf_matrix(term_document_matrix):
  '''Given the term document matrix, output a tf-idf weighted version.
  See section 15.2.1 in the textbook.
  
  Hint: Use numpy matrix and vector operations to speed up implementation.
  Input:
    term_document_matrix: Numpy array where each column represents a document 
    and each row, the frequency of a word in that document.
  Returns:
    A numpy array with the same dimension as term_document_matrix, where
    A_ij is weighted by the inverse document frequency of document h.
  '''

  # YOUR CODE HERE
  idf = []
  rows = term_document_matrix.shape[0]
  N = term_document_matrix.shape[1]
  print('N is: ',N)
  print('rows are: ',rows)
  
  for row in range(rows):
    #print(row,np.log10(N/(np.count_nonzero(term_document_matrix[row,:]))))
    idf.append( np.log10(N/(np.count_nonzero(term_document_matrix[row,:]))))
    

  term_document_matrix = 1 + np.log10(term_document_matrix)
  term_document_matrix = term_document_matrix.clip(min=0)

  for row in range(rows):
    term_document_matrix[row,:] = term_document_matrix[row,:] * idf[row]

  return term_document_matrix

def compute_cosine_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.
  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array
  Returns:
    A scalar similarity value.
  '''
  numerator = vector1.dot(vector2)
  den_1 = np.linalg.norm(vector1)
  den_2 = np.linalg.norm(vector2)
  
  # YOUR CODE HERE
  return (numerator) / (den_1 * den_2 )

def compute_jaccard_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.
  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array
  Returns:
    A scalar similarity value.
  '''
  
  # YOUR CODE HERE
  num = np.sum ( np.minimum( vector1, vector2 ) )
  den =  np.sum( np.maximum( vector1, vector2 ) )
  if( den == 0 or num == 0):
    return den
  
  return num / den

def compute_dice_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.
  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array
  Returns:
    A scalar similarity value.
  '''

  # YOUR CODE HERE
  j = compute_jaccard_similarity(vector1,vector2)
  
  return 2*j/(1+j)

def rank_plays(target_play_index, term_document_matrix, similarity_fn):
  ''' Ranks the similarity of all of the plays to the target play.
  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.
  Inputs:
    target_play_index: The integer index of the play we want to compare all others against.
    term_document_matrix: The term-document matrix as a mxn numpy array.
    similarity_fn: Function that should be used to compared vectors for two
      documents. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.
  Returns:
    A length-n list of integer indices corresponding to play names,
    ordered by decreasing similarity to the play indexed by target_play_index
  '''
  
  # YOUR CODE HERE
  N = len(term_document_matrix)
  V = len(term_document_matrix[0])

  target_play_vector = get_column_vector(term_document_matrix, target_play_index)

  sim_dict = {}

  for i in range(V):
      if i != target_play_index:
          other_play_vector = get_column_vector(term_document_matrix, i)
          sim_score = similarity_fn(target_play_vector,other_play_vector)
          sim_dict[i] = sim_score

  ranked = sorted(sim_dict.items(), key=lambda x: x[1], reverse = True)
  result = []
  for index,v in ranked:
      result.append(index)
  return result

def rank_words(target_word_index, matrix, similarity_fn):
  ''' Ranks the similarity of all of the words to the target word.
  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.
  Inputs:
    target_word_index: The index of the word we want to compare all others against.
    matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.
    similarity_fn: Function that should be used to compared vectors for two word
      ebeddings. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.
  Returns:
    A length-n list of integer word indices, ordered by decreasing similarity to the 
    target word indexed by word_index
  '''

  # YOUR CODE HERE
  N = len(matrix)

  target_word_vector = get_row_vector(matrix, target_word_index)

  sim_dict = {}
  for i in range(N):
      if i != target_word_index:
          other_play_vector = get_row_vector(matrix, i)
          sim_score = similarity_fn(target_word_vector, other_play_vector)
          sim_dict[i] = sim_score

  ranked = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
  result = []
  for index, v in ranked:
      result.append(index)
  return result

if __name__ == '__main__':
  tuples, document_names, vocab = read_in_shakespeare()

  print('Computing term document matrix...')
  td_matrix = create_term_document_matrix(tuples, document_names, vocab)

  print('Computing tf-idf matrix...')
  tf_idf_matrix = create_tf_idf_matrix(td_matrix)

  print('Computing term context matrix...')
  tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=2)

  print('Computing PPMI matrix...')
  PPMI_matrix = create_PPMI_matrix(tc_matrix)

  random_idx = random.randint(0, len(document_names)-1)
  similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar plays to "%s" using %s are:' % (document_names[random_idx], sim_fn.__qualname__))
    ranks = rank_plays(random_idx, td_matrix, sim_fn)
    for idx in range(0, 10):
      doc_id = ranks[idx]
      print('%d: %s' % (idx+1, document_names[doc_id]))

  word = 'juliet'
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar words to "%s" using %s on term-context frequency matrix are:' % (word, sim_fn.__qualname__))
    ranks = rank_words(vocab_to_index[word], tc_matrix, sim_fn)
    for idx in range(0, 10):
      word_id = ranks[idx]  
      print('%d: %s' % (idx+1, vocab[word_id]))

  word = 'juliet'
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar words to "%s" using %s on PPMI matrix are:' % (word, sim_fn.__qualname__))
    ranks = rank_words(vocab_to_index[word], PPMI_matrix, sim_fn)
    for idx in range(0, 10):
      word_id = ranks[idx]
      print('%d: %s' % (idx+1, vocab[word_id]))
