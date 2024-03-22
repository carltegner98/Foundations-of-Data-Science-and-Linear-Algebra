**Short Summary**
This project, developed by me and two classmates for the Foundations of Data Science course, showcases our understanding and application of programming concepts along with linear algebra principles. 
The project is divided into three distinct parts, each targeting a specific area: implementation of basic array operations, encoding and decoding with Hamming's code, and computing text document similarity.

**Part 1: SNumPy - Simplified NumPy Implementation**
Our first task was to create a simplified version of the popular NumPy library, which we named SNumPy. 
This custom implementation supports basic functionalities such as array creation, reshaping, and arithmetic operations. Our focus was on demonstrating an understanding of array manipulation and operation without relying on NumPy's built-in functions.

**Features**
Creation of arrays filled with zeros (snp.zeros) and ones (snp.ones)
Reshaping arrays with snp.reshape
Array shape retrieval using snp.shape
Array concatenation through snp.append
Basic arithmetic operations: addition, subtraction, and dot product

**Part 2: Hamming's Code Encoder and Decoder**
The second part of the project involved implementing an encoder and decoder for Hamming's (7,4) code, a linear error-correcting code. 
Our implementation can encode a 4-bit message into a 7-bit codeword and decode it back, with capabilities to correct single-bit errors and detect two-bit errors.

**Functionality**
Encoding 4-bit binary values into 7-bit codewords
Decoding 7-bit codewords to retrieve the original 4-bit messages
Error detection and correction based on Hamming's algorithm

**Part 3: Text Document Similarity**
Our final task was to develop a program to compute the similarity between text documents. 
This involved creating a dictionary of words from a given text corpus and then computing similarity scores between documents using vector space models.

**Approach**
Construction of a word dictionary from the text corpus
Transformation of text documents into word vectors
Computation of document similarity using dot product and Euclidean distance measures
