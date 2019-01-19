

# Data cleaning
# ===========================================================================
import re
import pandas as pd
def clean_text(text):
    # replace  . and a space with only a space, then amke all words lower case.
    text = text.replace(". "," ").replace(",","").lower()
    # get rid of the . at the end of each line. 
    cleaned_text = re.sub("\.$","",text)
    
    return cleaned_text
 


class text_clean:
    """
    A class to help with cleaning text data. 
    """
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        assert isinstance(X,pd.Series), "The input data should be pandas Series."
        X = X.apply(clean_text)
        
        return X


# Word embedding training 
# ===========================================================================
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.externals import joblib

from sklearn.base import BaseEstimator

def _find_part_pii(text, model, sep = " "):
    tokenized_text = text.split(sep)
    
    part_pii = model.wv.doesnt_match(tokenized_text)
    
    return part_pii    



def _extracted_pii2matrix(pii_list, model):
    # set the matrix dimensions
    column_num = model.trainables.layer1_size
    row_num = len(pii_list)
    # initialized the matrix
    pii2vec_mat = np.zeros((row_num, column_num))
    # iterate through the pii_list and assign the vectors to matrix.
    for index, ith_pii in enumerate(tqdm(pii_list)):
        pii2vec_mat[index,:] = model.wv[ith_pii]
    
    return pii2vec_mat



class word_embedding(BaseEstimator):
    """
    A class to convert words/docs to vectors by applying any model supported by gensim.  
    
    This class will allow continued training on the pre-trained model by assigning
    the model to the pre_trained option in class initialization.  
    
    After training the model, it will dump the word2vec model to the path assigned to 
    dump_file option.  
    
    
    """
    def __init__(self, algo_name = "word2vec", size = 100, min_count = 1, window = 5, workers =1,\
                 epochs = 5, pre_train = None, dump_file = False, continue_train_pre_train = True,
                 re_train_new_sentences = True):
        
        
        assert algo_name in ["word2vec", 'fasttext', 'doc2vec'], \
        "please enter a model name in ['word2vec', 'fasttext', 'doc2vec']"
        
        self.algo_name = algo_name
        self.epochs = epochs 
        self.pre_train = pre_train
        self.dump_file = dump_file 
        self.re_train_new_sentences = re_train_new_sentences
        self.continue_train_pre_train = continue_train_pre_train
        
        # model options
        self.size = size
        self.min_count = min_count
        self.window = window
        self.workers = workers
        
        
    def _algo_init(self):
        if self.algo_name == "word2vec":
            model = Word2Vec(size = self.size, min_count = self.min_count,
                            window = self.window, workers = self.workers)
        elif self.algo_name == "fasttext":
            model = FastText(size = self.size, min_count = self.min_count,
                            window = self.window, workers = self.workers)
        elif self.algo_name == "doc2vec":
            model = Doc2Vec(vector_size = self.size, min_count = self.min_count,
                            window = self.window, workers = self.workers)
            
        self.model = model
        return self

    def _embedding_training(self, sentences, update = False):
        """
        This helper functions will build the vocabulary, train the model and update the self.model
        with the newly trained model.
        
        if update = True, it will update the vocabulary and the model can continue to train.
        If update = False, the model will rebuild a new vocabulary from scratch using the input data.
        """
        updated_model_with_vocab = self.model

        updated_model_with_vocab.build_vocab(sentences, update = update)
        
        updated_model_with_vocab.train(sentences, total_examples = len(sentences), epochs = self.epochs)
        
        # update the model with the trained one. 
        self.model = updated_model_with_vocab
        
    def _pd_to_gensim_format(self, text):
        
        # special handling for doc2vec model. 
        if self.algo_name == "doc2vec":
            documents = [TaggedDocument(sentence.split(" "), [index])\
                          for index, sentence in enumerate(text)] 
            print("Using index for the tags")    
        else:
            documents = [sentence.split(" ") for sentence in text]
            
            
        return documents
            
        
    def fit(self, X, y = None):
        """
        The fit method will get use the pre_trained model if the model is assigned to the pre_train option.
        
        If the pre_train is None, then the model will be trained. 
        
        
        If the pre_train model is not None, then the default is to continue training on the new model. 
        Unless option continue_train_pre_train is specified as False. The False option will just assign 
        the pre_train model to self.model
        """
        gensim_X = self._pd_to_gensim_format(text = X)
        
        if self.pre_train is not None:
            
            # update the pre_trained model with new training data
            if self.continue_train_pre_train:
                self.model = self.pre_train
                self._embedding_training(sentences = gensim_X, update = True)
                print("continue training with the pre_train model.")
                
            # no training the pre_trained model. 
            elif not self.continue_train_pre_train:
                self.model = self.pre_train
                print("No training with pre_train model.")
                
            
            return self
        
        else:
            # initialize the model, split the sentence into tokens and train it. 
            self._algo_init()
            self._embedding_training(sentences = gensim_X)
            print("Building new vocabulary and training the {} model".format(self.algo_name))
        
        
        # dump the model to disk
        if isinstance(self.dump_file,str):
            self.model.save(self.dump_file)
            print("Writing the {} model to disk.".format(self.algo_name))
            
        return self
        
    
    def transform(self, X):
        """
        If re_train_new_sentences is True, which is the default setting, 
        the model will be re-trained on the new sentences. 
        This will create word embedding for words not in the original vocabulary.
        This will increase the model inference time since it invovles model training. 
        
        For using word2vec to predict PII data, it is recommended to update the model with new sentences. 
        For fastttext, it is not necessary since it will infer from the character n-grams. The fasttext training
        is much longer than word2vec. 
        """
        gensim_X = self._pd_to_gensim_format(text = X)
        
        # update the embedding with new sentences or train the model. 
        if self.re_train_new_sentences:
            self._embedding_training(sentences = gensim_X, update = True)
            print("transforming while training {} model with new data.".format(self.algo_name))
            
            
        # extract the PII 
        extracted_pii_list = [_find_part_pii(text = text, model = self.model)\
                    for text in tqdm(X) ]
        
        # convert the extracted pii text into vectors.
        piivec_matrix = _extracted_pii2matrix(pii_list = extracted_pii_list,\
                                          model = self.model)
        return piivec_matrix 
                                          
