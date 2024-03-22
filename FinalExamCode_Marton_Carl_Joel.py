#!/usr/bin/env python
# coding: utf-8

# In[12]:


############QUESTION1#############
class SNumPy:
    
## VALIDATION, CHECKING IF INPUT IS CORRECT
    def _validate_matrix(self, matrix):
        
        for ix, row in enumerate(matrix): #OUTER LOOP - iterates over each row of the matrix
                                          #Using enumerate to get both index & content of row
            for jx, other_row in enumerate(matrix): #INNER LOOP
                
                if ix != jx and len(row) != len(other_row): #check every row against every other row to ensure equal length
                    raise ValueError(f"Row {ix+1} length does not match row {jx+1}.")
                    
            for i, n in enumerate(row):
                if isinstance(n, str) and n.isnumeric():
                    matrix[ix][i] = int(n)
                    
                elif not (isinstance(n, int) or isinstance(n, float)): #
                    raise ValueError(f"Non-numeric value at row {ix+1}, column {i+1}") #using +1 here since i want the real position, not the index position
                    
        return matrix #if correct, it will return the matrix. in other words - if this function returns none, the matrix isnt valid

    def _validate_vector(self, vector): #VALIDATING VECTOR

        #checks for nested loops, if the user made a vector looking like this "[[1,1,1,1]]"
        if any(isinstance(n, list) for n in vector):
          raise ValueError("Nested list detected. Please remove the outer brackets. Example: [value, value, value, value]")

        #same logic as the matrix function
        for ix, n in enumerate(vector):
            
            if isinstance(n, str) and n.isnumeric():
                vector[ix] = int(n)
                
            elif not (isinstance(n, int) or isinstance(n, float)):
                raise ValueError(f"Non-numeric value at position {ix+1}")
                
        return vector #same as above

    #using the matrix and vector validation
    def input_check(self, array):
        try: #having try & except here instead using it in validate matrix/vector functions
            
            if len(array) > 1 and all(isinstance(row, list) for row in array): #if rows are more than one, and all rows are list = matrix
                return self._validate_matrix(array)
            
            else:  #assuming its a vector otherwise
                return self._validate_vector(array)
        
        except ValueError as e:
            print(f"An error occurred: {e}")
            return None  # Or handle the error as needed


#TASK 1
#snp.ones(Int): the ones function takes an int parameter and returns an array (list) of length int
#parameter, and the array contains only ones. Example: snp.ones(5) = [1,1,1,1,1]
    def ones(self, length):
        try:
            
            if not isinstance(length, int):
                raise TypeError("Input must be an integer")
            
            if length < 0:
                raise ValueError("Length must be positive number")
                
            list = [1] * length
            return list
        
        except (ValueError, TypeError) as e:
          print(f"An error ocurred: {e}")
          return None

#TASK 2
#snp.zeros(Int): similar to the ones function, expect returns an array of zeros instead of ones.
    def zeros(self, length):
        try:
            
            if not isinstance(length, int):
                raise TypeError("Input must be an integer")
                
            if length < 0:
                raise ValueError("Length must be positive number")
                
            list = [0] * length
            return list
        
        except (ValueError, TypeError) as e:
          print(f"An error ocurred: {e}")
          return None


#TASK 3
#snp.reshape(array, (row, column)): takes an array and converts it into the dimensions specified
#by the tuple (row, column). Hence this function converts from a vector to a matrix. For an
#example on reshape functionality of numpy, refer to fig. 1.

    def reshape(self, array, dimensions):
        try:

            #validations
            validated_array = self.input_check(array)
            if validated_array is None: #since our validate functions returns none if the matrix/vector isnt valid
                return None

            if isinstance(validated_array[0], list): #chceking
               raise ValueError("Reshaping is only allowed for one-dimensional arrays (vectors)")

            #checking dimensions
            if not isinstance(dimensions, tuple) or len(dimensions) != 2:
                raise ValueError("Dimensions must be a tuple of two integers")

            row, column = dimensions

            if not isinstance(row, int) or not isinstance(column, int):
                raise TypeError("Row and column values must be integers")

            if len(validated_array) != row * column:
                raise ValueError(f"Number of elements in the array ({len(validated_array)}) does not match the specified dimensions ({row} x {column} = {row*column}).")

            reshaped_array = [validated_array[i * column:(i + 1) * column] for i in range(row)]

            print(f"Transformation completed: The vector has been successfully reshaped into a {row} x {column} matrix.")

            return reshaped_array

        except (ValueError, TypeError) as e:
          print(f"An error ocurred: {e}")
          return None

#TASK 4
#snp.shape(array) : returns a tuple with the matrix/vector’s dimension e.g. (# rows, # columns)

    def shape(self, array):

          #empty array
          if len(array) == 0:
              return 0, 0

          # Validate the input array
          array = self.input_check(array)
            
          if array is None:
              return None  # Validation failed

          #matrix or vector`?
          if all(isinstance(row, list) for row in array):  # Matrix
              return len(array), len(array[0])
          else:  # Vector
              return 1, len(array)


#TASK 5
#snp.append(array1, array2) : returns a new vector/matrix that is the combination of the two
#input vectors/matrices. Note that you can’t append a vector to a matrix and vice versa and
#therefore use suitable exception handling and throw/return user friendly error messages.

    def append(self, array1, array2):
      try:

          array1 = self.input_check(array1)
          array2 = self.input_check(array2)

          if array1 is None or array2 is None:
              return None

          #shape function to get dimensions of the arrays
          shape1 = self.shape(array1)
          shape2 = self.shape(array2)

          #matrix check
          if shape1[1] > 1 or shape2[1] > 1:
              #If array2 is a vector but has the same number of elements as the columns of array1, treat it as a single row matrix
              if shape1[1] == len(array2) and isinstance(array2[0], (int, float)):
                  array2 = [array2]  #wrap the vector in a list to make it a single row matrix, otherwise it gives us the wrong input

              if shape1[1] == shape2[1]:
                  return array1 + array2
              else:
                  raise ValueError("Matrices must have the same number of columns to be appended")
          else:
              #both are vectors
              return array1 + array2

      except ValueError as e:
          print(f"An error occurred: {e}")
          return None




#TASK 6
#snp.get(array, (row, column)): returns the value specified by the coordinate point (row, column)
#of the array provided (can be vector or matrix).

    def get(self, array, coordinates):
        try:
            #validations
            array = self.input_check(array)
            if array is None:
                return None  # Validation failed

            #checking coordinates
            if not isinstance(coordinates, tuple) or len(coordinates) != 2:
                raise ValueError("Coordinates must be a tuple of two integers")

            row, column = coordinates

            #must be positive value
            if row < 0 or column < 0:
                raise IndexError("Row and column indices must be non-negative")

            #because index starts at 0
            row -= 1
            column -= 1

            ##dimensions using the shape function
            rows, cols = self.shape(array)

            #vectors
            if cols == 1:  # Assuming vectors are treated as single-column arrays
                if column >= cols:
                    raise IndexError(f"Index out of range for the vector. Vector length is {cols}.")
                return f"The value at the chosen coordinate ({coordinates[0]}, {coordinates[1]}) is {array[column]}."

            #matrices
            if row >= rows or column >= cols:
                raise IndexError(f"Index out of range for the matrix. Matrix dimensions are {rows}x{cols}.")
            return f"The value at the chosen coordinate ({coordinates[0]}, {coordinates[1]}) is {array[row][column]}."

        except (ValueError, TypeError, IndexError) as e:
            print(f"An error occurred: {e}")
            return None



#TASK 7
#snp.add(array1, array1) : addition on vectors/matrices.

    def add(self, array1, array2):
        try:

            array1 = self.input_check(array1)
            array2 = self.input_check(array2)

            if array1 is None or array2 is None:
                return None  #validation failed

            #using shape function here aswell
            shape1 = self.shape(array1)
            shape2 = self.shape(array2)

            #same dimensions?
            if shape1 != shape2:
                raise ValueError("Both arrays must have the same dimensions")

            #if all instances are float or int, its a vector, if its a matrix it should containt list
            if all(isinstance(el, (int, float)) for el in array1) and all(isinstance(el, (int, float)) for el in array2):
                return [x + y for x, y in zip(array1, array2)]

            #addition for matrices (2D arrays)
            return [[x + y for x, y in zip(row1, row2)] for row1, row2 in zip(array1, array2)]
        
        except ValueError as e:
            print(f"An error occurred: {e}")
            return None


#TASK 8
#snp.subtract(array1, array1) : subtraction on vectors/matrices.
    def subtract(self, array1, array2):
        try:

            array1 = self.input_check(array1)
            array2 = self.input_check(array2)

            if array1 is None or array2 is None:
                return None  #vlidation failed

            shape1 = self.shape(array1) # same logic as the add function
            shape2 = self.shape(array2)

            if shape1 != shape2:
                raise ValueError("Both arrays must have the same dimensions")

            if all(isinstance(el, (int, float)) for el in array1) and all(isinstance(el, (int, float)) for el in array2):
                return [x - y for x, y in zip(array1, array2)]

            return [[x - y for x, y in zip(row1, row2)] for row1, row2 in zip(array1, array2)]
        
        except ValueError as e:
            print(f"An error occurred: {e}")
            return None
#TASK9
#snp.dotproduct(array1, array1): computes the dot product between two arrays (which could be
#vectors or/and matrices) and returns an appropriate value. Use appropriate exception handling
#to output user-friendly error messages if the dot product cannot be performed between the
#given arrays.
    def dotproduct(self, array1, array2):
        try:

            array1 = self.input_check(array1)
            array2 = self.input_check(array2)

            if array1 is None or array2 is None:
                return None

            #using shape function here aswell
            shape1 = self.shape(array1)
            shape2 = self.shape(array2)
            #since shape function returns (x, y) we can use this to check if rows equals columns etc
            if shape1[1] == 1 and shape2[1] == 1:  # Both are vectors
                
                if shape1[0] != shape2[0]:
                    raise ValueError("Vectors must be of the same length for dot product")
                return sum(x * y for x, y in zip(array1, array2))

            #matrix
            elif shape1[1] > 1 and shape2[1] > 1:  #both are matrix
                if shape1[1] != shape2[0]:
                    raise ValueError("For a dot product, the number of columns in the first matrix must match the number of rows in the second matrix, and vice versa.")

                result = []
                for i in range(len(array1)):
                    result_row = []
                    for j in range(len(array2[0])):
                        sum = 0
                        for k in range(len(array1[0])):
                            sum += array1[i][k] * array2[k][j]
                        result_row.append(sum)
                    result.append(result_row)
                return result

            else:
                raise ValueError("Dot product can only be performed between two vectors or two matrices")

        except ValueError as e:
            print(f"An error occurred: {e}")
            return None

#[Optional] If you are not challenged enough by the above questions, then implement a solver
#for a system of linear equations using Gaussian elimination and row reduction rules for the functionality as depicted in https://docs.scipy.org/doc/numpy/reference/generated/numpy.
#linalg.solve.html.

    def gaussian_elimination(self, A, b, decimal_places=2): #
        try:

            #validation
            A = self.input_check(A)
            b = self.input_check(b)
            if A is None or b is None:
                return None

            #using shape function to check dimensions
            shape_A = self.shape(A)
            if shape_A[0] != shape_A[1]:
                raise ValueError("Matrix A must be square.")
            if len(b) != shape_A[0]:
                raise ValueError("Length of vector b must match the number of rows in matrix A.")

            n = len(A)  #n is the number of rows in A (and the length of b)

            #augmented matrix
            for i in range(n): #for every row, plus the corresponding element
                A[i] = A[i][:] + [b[i]]

            #Gaussian Elimination
            for i in range(n): #looping through each row of the augmented matrix

                max_el = abs(A[i][i]) #find the absolute value(furthest away from 0)
                max_row = i
                #this loop checks each row below the current row 'i' to find the row
                #with maximum absolute value in the 'column'
                for k in range(i+1, n):
                    if abs(A[k][i]) > max_el: #This condition checks if the absolute value..
                                              #of the element in the k row and i column is greater than the current maximum (max_el).
                        max_el = abs(A[k][i])
                        max_row = k

                if A[i][i] == 0:
                  print("The system does not have a unique solution.")
                  return None


                #if a row with large abs value in the i column is found, this loop
                #swaps that row with the current row 'i'
                for k in range(i, n+1):
                    A[max_row][k], A[i][k] = A[i][k], A[max_row][k]

                #this one loop iterates every row below current row 'i'
                for k in range(i+1, n):
                    c = -A[k][i]/A[i][i] #used to make the current column element zero
                    for j in range(i, n+1): #applies the c to each element of row k from column i to the end
                        if i == j:
                            A[k][j] = 0
                        else:
                            A[k][j] += c * A[i][j]

            #this goes backward, from last to first, calculating the value of i variable and store it in x[i]
            #the inner loop updates the augmented part of all rows, above the current row i to refect the value of x[i]
            x = [0 for i in range(n)]
            for i in range(n-1, -1, -1):
                x[i] = A[i][n]/A[i][i]
                for k in range(i-1, -1, -1):
                    A[k][n] -= A[k][i] * x[i]

            #rounding because of floating-point arithmetic inaccuracies inherent in computer calculations
            rounded_x = [round(num, decimal_places) for num in x]

            return rounded_x  # Return the solution vector

        except ValueError as e:
            print(f"An error occurred: {e}")
            return None


# In[21]:


############QUESTION1#############
snp = SNumPy()

#1
one = snp.ones(4)

#2
zero = snp.zeros(4)

#3
one_new = snp.reshape(one, (2,2))
zero_new = snp.reshape(zero, (2,2))

#4
snp.shape(one_new)

#5
snp.append(one_new, zero_new)

#6
three_four = [[9,7,3,2], [1,10,11,12], [4, 5, 6, 8]]
snp.get(three_four, (2, 3))

#7
one_addition = snp.add(one_new, one_new)
print(one_addition)

#8
one_subtraction = snp.subtract(one_new, one_new)
print(one_subtraction)

#9
matrice1 = [[1,3,4], [1,3,2], [1,3,3], [5,4,3]]
matrice2 = [[1,3,4,4], [5,4,3,4], [3,2,1,4]]
snp.dotproduct(matrice1, matrice2)
            
#10
A = [[1,0,-2], [-2,1,6], [3,-2,-5]]
b = [-1,7,-3]
snp.gaussian_elimination(A, b)



# In[26]:


############QUESTION2#############

import numpy as np
import random as r
import copy

#Generator matrix for Hamming(7,4) code
G = [[1, 1, 0, 1],
     [1, 0, 1, 1],
     [1, 0, 0, 0],
     [0, 1, 1, 1],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]

#Parity-check matrix for Hamming(7,4) code
H = [[1, 0, 1, 0, 1, 0, 1],
     [0, 1, 1, 0, 0, 1, 1],
     [0, 0, 0, 1, 1, 1, 1]]

#Decoder Matrix for Hamming(7,4) code
R = [[0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 1]]

def hamming_encoder(input):
    #Ensure the input is a 4-bit binary vector
    if len(input) != 4:
        raise ValueError("Data input must be 4-bit")
    elif any(element not in {0, 1} for element in input):
        raise ValueError("Input must be binary")

    #Encode the input using the generator matrix
    codeword = np.dot(G, input)
    codeword = np.mod(codeword, 2)  # Ensure binary output

    return codeword

def parity_check(codeword):
    #Convert codeword to a NumPy array
    codeword = np.array(codeword)

    # Compute parity using the parity-check matrix
    parity = np.mod(np.dot(H, codeword), 2)

    #Determine the index from the binary vector
    errorvalue = np.sum(parity * [1, 2, 4])

    #Fix one-bit error in the codeword if detected
    if sum(parity) != 0:
        codeword[errorvalue - 1] = 1 - codeword[errorvalue - 1]
        return f"Error in codeword: {parity}, correct codeword: {codeword}"
    else:
        return f"No error in codeword: {parity}, correct codeword: {codeword}"

def hamming_decoder(codeword):
    #Check if input codeword is a 7-bit vector
    if len(codeword) != 7:
        raise ValueError("Input codeword must be a 7-bit binary vector.")
    elif any(element not in {0, 1} for element in codeword):
        raise ValueError("Input codeword must be a binary vector.")

    #Decode the codeword using matrix-vector product
    decoded_data = np.mod(np.dot(R, codeword), 2)

    return decoded_data

def test(input, error):
    k = r.randint(0, 6)  #Randomly choose an index from 0 to 6
    codeword = hamming_encoder(input)  # Encode the input into a codeword

    if error == 0: #No errors it only returnes the output of previous functions
        return f"Input value: {input}\nCodeword: {hamming_encoder(input)}\n{parity_check(codeword)}\nWord is: {hamming_decoder(codeword)}"

    elif error == 1: #If one error, one bit is randomly flipped to 1 if it was 0, and vice versa
        if codeword[k] == 1:
            codeword[k] = 0
        elif codeword[k] == 0:
            codeword[k] = 1

        #Identify error and correct it
        parity = np.mod(np.dot(H, codeword), 2)
        c_copy = copy.deepcopy(codeword)  #Create a deep copy to avoid modifying the original for the output
        errorvalue = np.sum(parity * [1, 2, 4])
        c_copy[errorvalue - 1] = 1 - c_copy[errorvalue - 1]

        return f"Input value: {input}\nDistorted codeword: {codeword}" \
               f"\n{parity_check(codeword)}\nWord is: {hamming_decoder(c_copy)}"

    elif error == 2:
        for _ in range(2):
            idx = r.randint(0, 6)  #Choose a random index from 0 to 6
            codeword[idx] = 1 - codeword[idx]  #Flip 
            if codeword[k] == 1:
                codeword[k] = 0
            elif codeword[k] == 0:
                codeword[k] = 1

        #Returns what the input value is, what the distorted codeword is, what the actual correct codeword is.
        #It then returns the output for the Parity_check function to show that 2-bit error are not actually corrected.
        #Finally returns what an incorrect word, again, illustrating that 2-bit errors can not be corrected.
        return f"Input value: {input}\nDistorted codeword: {codeword}\nCorrect codeword:{hamming_encoder(input)}," \
               f"\n{parity_check(codeword)}\nThe decoded message shows an incorrect word:{hamming_decoder(codeword)} "

#Provides examples of 0, 1, and 2 errors
w = (1,0,0,1)
print(test(w,0),"\n")
print(test(w,1), "\n")
print(test(w,2))


# In[2]:


############QUESTION3#############

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import os

def read_file(file_path):
    #Reads a file and returns its content. Supports both PDF and text files.
    _, file_extension = os.path.splitext(file_path)
    try:
        if file_extension.lower() == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def vocab_vect(corpus):
    #Creates a vocabulary for the given corpus for document vectorization.
    token_pattern = r'(?u)\b\w+\b|\b\d{4}\b'
    vectorizer = CountVectorizer(token_pattern=token_pattern, binary=True)
    vectorizer.fit(corpus)
    return vectorizer

def vectorize_docs(corpus, vectorizer):
    #Converts the corpus into a vectorized form using the vectorizer from 'vocab_vect'.
    return vectorizer.transform(corpus)

def compute_similarity(vector, vectors):
    #Computes the cosine similarity between the given vector and a set of vectors.
    return cosine_similarity(vector, vectors)

def main(document_paths, search_document_path):
    #Main function to process the documents and compute similarity scores.
    if not isinstance(document_paths, list) or not all(isinstance(path, str) for path in document_paths):
        raise ValueError("Corpus must be a list of strings.")
    if not isinstance(search_document_path, str):
        raise ValueError("Search document path must be a string.")

    search_document = read_file(search_document_path)
    if not search_document:
        raise ValueError("Search document is empty or could not be read.")

    corpus = [read_file(path) for path in document_paths]
    
    # Filter out unreadable documents
    unreadable_docs = []
    readable_corpus = []
    readable_docs = []

    for doc, path in zip(corpus, document_paths):
        if doc is None:
            unreadable_docs.append(path)
        else:
            readable_corpus.append(doc)
            readable_docs.append(path)

    if unreadable_docs:
        unreadable_docs_str = ", ".join(unreadable_docs)
        print(f"The following document(s) could not be read and will be excluded: {unreadable_docs_str}")

    # Proceed with the readable documents
    if not readable_corpus:
        print("No readable documents in the corpus.")
        return []

    vectorizer = vocab_vect(readable_corpus)
    corpus_vectors = vectorize_docs(readable_corpus, vectorizer)
    search_vector = vectorize_docs([search_document], vectorizer)
    similarity = compute_similarity(search_vector, corpus_vectors)

    # Pair each document's path with its similarity score and sort them
    scores = sorted(((path, score) for path, score in zip(readable_docs, similarity.flatten())), key=lambda x: x[1], reverse=True)
    return [(os.path.basename(path), score) for path, score in scores]

def get_file_title(file_path):
    #Extracts the file title from a file path.
    base_name = os.path.basename(file_path)
    return os.path.splitext(base_name)[0]

def run_test(corpus_path, search_document_path):
    # Check if the list of search document paths is empty
    if not search_document_path:
        print("No search documents provided.")
        return
    
    #Run the main function and print the sorted documents.
    for search_path in search_document_path:
        print(f"\nTesting with search document: {get_file_title(search_path)}")
        try:
            sorted_doc_scores = main(corpus_path, search_path)
            for i, (title, _) in enumerate(sorted_doc_scores, start=1):
                print(f"{i}: {title}")
        except ValueError as e:
            print(f"Error: {e}")


corpus_path = []   

search_document_path = []

run_test(corpus_path, search_document_path)


# In[ ]:




