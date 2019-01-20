# piidetect
A package to build an end-to-end ML pipeline to detect personally identifiable information (PII) from text. This 
package is still in early stage development. More documentations and tests are coming soon. 

# installation
```
pip install piidetect
```

## Create fake PII
fakepii.py is the module to create random text mixed with different types of PII.

### Use in Python

Creating fake text in Python
```
from piidetect.fakepii import Fake_PII
fake_ = Fake_PII()
fake_.create_fake_profile(10)
train_labels, train_text, train_PII = fake_.create_pii_text_train(n_text = 5)
```

This package also has some helper functions to create fake pii with text and dump it to disk. 

```
from piidetect.fakepii import Fake_PII, write_to_disk_train, write_to_disk_test

write_to_disk_train(10)
write_to_disk_test(20)
```
The file name for training data will be "train_text_with_pii_" + convert_datetime_underscore(datetime.now()) + ".csv"
The file name for testing data will be "test_text_with_pii_" + convert_datetime_underscore(datetime.now()) + ".csv"

The dumped data will contain three columns: "Text", "Labels", "PII".
The Text column contains the text mixed with PII.
The Labels column contains the PII type of the text. If there is no PII in the text, then it is "None".
The PII column contains the True PII. 

### Command line usage
You can just download the fakePII.py to your local directory to use with command line. 
Here are some examples for command line usage.
```
# creating 1000 training data and 100 testing data. 
python fakePII.py -train 1000 -test 100
# creating 100 testing data
python fakePII.py  -test 100
# create 1000 training data
python fakePII.py -train 1000 
```

In the training text, a normal text is repeated used to insert different PIIs into
it. In the testing text, a normal text is not intentionally repeated to insert different PIIs. 


## Word embedding training
This package wraps the word embedding algorithm **word2vec, doc2vec and fasttext** for detecting PII. 

This word_embedding will allow continued training on the pre-trained model by assigning
the model to the **pre_trained** option in class initialization.  

After training the model, it will dump the word2vec model to the path assigned to 
**dump_file** option (can not dump to a path if the directory does not exist)

If the **pre_train** is None, then the model will be trained. 

If the **pre_train** model is not None, then the default is to continue training on the new model
unless option **continue_train_pre_train** is specified as False. The False option will just assign 
the pre_train model to be the model without training on the text. 

If **re_train_new_sentences** is True, which is the default setting, the model will be re-trained on the new sentences. 
This will create word embedding for words not in the original vocabulary.
This will increase the model inference time since it invovles model training. 
        
For using word2vec to predict PII data, it is recommended to update the model with new sentences. 
For fastttext, it is not necessary since it will infer from the character n-grams. The fasttext training
is much longer than word2vec. 

**size**: vector dimension for word. Must be the same as the pre_train model is that is specified.

**min_count**: Ignores all words with total frequency lower than this. Use 1 for PII detection.

**workers**: number of CPU cores for training


```
from piidetect.pipeline import word_embedding
model = word_embedding(algo_name = "word2vec",size = 100, min_count = 1, workers =2)
model.fit(data['Text'])
```
## How to use piidetect to build a pipeline for PII detection. 
Before you start to train an end-to-end PII detector, you need to create binary labels 
for ML models.
```
from piidetect.pipeline import binary_pii
data['Target'] = data['Labels'].apply(binary_pii)
```

This is an example in building an end-to-end PII detection with logistic regression. 
```
from piidetect.pipeline import word_embedding, text_clean
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

logit_clf_word2vec = LogisticRegression(solver = "lbfgs", max_iter = 10000)

word2vec_pipe = Pipeline([('text_cleaning', text_clean()),
                 ("word_embedding", word_embedding(algo_name = "word2vec", workers =2)),
                 ("logit_clf_word2vec",logit_clf_word2vec)
                ])
                
word2vec_pipe.fit(data["Text"],data['Target'] )
```
You can also use RandomizedSearchCV to hyperparameter selection. (This is going to run for a long time.)
```
from sklearn.model_selection import RandomizedSearchCV
from piidetect.pipeline import word_embedding, text_clean
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


logit_clf_word2vec = LogisticRegression(solver = "lbfgs", max_iter = 10000)

pipe = Pipeline([('text_cleaning', text_clean()),
                 ("word_embedding", word_embedding( workers =2)),
                 ("logit_clf_word2vec",logit_clf_word2vec)
                ])


param_grid = {
    'word_embedding__algo_name':['word2vec', 'doc2vec','fasttext'],
    'word_embedding__size':[100,200,300],   
    'logit_clf_word2vec__C': uniform(0,10),
    'logit_clf_word2vec__class_weight':[{0: 0.9, 1: 0.1}, {0: 0.8, 1: 0.2}, {0: 0.7, 1: 0.3},None]
}

pipe_cv = RandomizedSearchCV(estimator = pipe,param_distributions = param_grid,\
                                      cv =10, error_score = 0,n_iter = 10 , scoring = 'f1'\
                                      ,return_train_score=True, n_jobs = 1)
```


You can dump the pipeline to disk after training. The compress = 1 will save the pipeline into one file. 
For a model with size = 300 with word2vec, the model can be around 1GB. 

```
from sklearn.externals import joblib
joblib.dump(pipe_cv.best_estimator_, 'pipe_cv.pkl', compress = 1)

```

