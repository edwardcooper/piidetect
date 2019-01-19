# for command line usage
import sys, getopt


# for creating fake values
from faker import Faker
# for random sampling
from numpy.random import choice as choose
from numpy.random import seed as setseed
# for progress bar
from tqdm import tqdm
# for generating file name with current time
from datetime import datetime
# for saving the final dataframe.
import pandas as pd




def sep_change(text, init_sep, after_sep):
    """
    A function to separate the input text into strings by the init_sep variable
    
    and combine the strings again by the after_sep variable.
    """
    splitted_strings = text.split(init_sep)
    
    combined_strings = after_sep.join(splitted_strings)
    
    return combined_strings

def convert_datetime_underscore(data):
    """
    A Helper function to convert all separators in datetime.now() into underscore. 
    """
    now_string = str(data)
    
    now_string = sep_change(now_string, init_sep = "-", after_sep = "_")
    
    now_string = sep_change(now_string, init_sep = " ", after_sep = "_")
    
    now_string = sep_change(now_string, init_sep = ":", after_sep = "_")
    
    now_string = sep_change(now_string, init_sep = ".", after_sep = "_")
    
    return now_string
    
def _random_sep_change(data, init_sep = "-", after_sep = " ", percentage = 0.5 , seed = 7):
    """
    A function to randomly change the SSN data's separator. 
    
    The input data is a list.
    """
    setseed(seed)
    # generate the index for replacing separator.
    replacing_indexes = choose(range(len(data)), int(len(data)*percentage))
    
    for each_replacing_index in replacing_indexes:
        # change the ssn data's separator from init_sep to after_sep
        data[each_replacing_index] = sep_change(data[each_replacing_index], init_sep, after_sep)
        
    return data
        

    
    

class Fake_PII():
    '''
    A class to generate a number of fake profiles, training/testing text mixed with
    different types of fake PIIs.

    Examples
    --------
    fake_ = Fake_PII()
    fake_.create_fake_profile(10)
    train_labels, train_text, train_PII = fake_.create_pii_text_train(n_text = 5)
    '''
    def __init__(self, n_profile = None,fake_profiles = None, seed = 7,\
                pii_with_text = None, pii_labels = None, PII = None):

            
        
        # initialize the Faker from faker package for fake data generation.
        try:
            self.faker = Faker()
        except ImportError as error:
            print(error.__class__.__name__ + ": " + error.message)
            

        self.n_profile = n_profile
        self.pii_with_text = pii_with_text
        self.pii_labels = pii_labels
        self.fake_profiles = fake_profiles
        self.seed = seed 
        self.PII = PII 
 
        
    def create_fake_profile(self, n_profile, verbose = False, ssn_sep_change = True):
        
        assert isinstance(n_profile, int), "Please enter an integer\
        for the number of generated profiles."
        self.n_profile = n_profile
        
        fake_profiles = dict()
        # use faker package to generate either a full/last name/first name.
        fake_profiles["Name"] = [choose([self.faker.name(),\
                                         self.faker.last_name(),
                                         self.faker.first_name()])\
                                 for _ in range(self.n_profile)]
        # use faker to generate either a full/secondary/street address
        fake_profiles["Address"] = [choose([self.faker.address(),\
                                            self.faker.street_address(),\
                                            self.faker.secondary_address()])\
                                   for _ in range(self.n_profile)]
        
        fake_profiles["SSN"] = [self.faker.ssn() for _ in range(self.n_profile)]
        
        fake_profiles["Email"] = [self.faker.email() for _ in range(self.n_profile)]
                                 
        fake_profiles["Plates"] = [self.faker.license_plate()\
                                   for _ in range(self.n_profile)]
                                        
        
        fake_profiles["CreditCardNumber"] = [self.faker.credit_card_number()\
                                             for _ in range(self.n_profile)]
                                     
        
        fake_profiles["Phone_number"] = [self.faker.phone_number()\
                                         for _ in range(self.n_profile)]
        
        # change the separator in SSN data.
        if ssn_sep_change:
            fake_profiles["SSN"] = _random_sep_change(fake_profiles["SSN"])
                                        
        # change the separator in Address data from "/n" to " "
        fake_profiles['Address'] = [sep_change(each_address, init_sep = "\n" , after_sep = " ")\
                                   for each_address in fake_profiles['Address'] ]
                                              
        
        
        self.fake_profiles = fake_profiles
        print("Finished creating fake profiles.")
        
        if verbose:
            return self.fake_profiles 
        
    def _init_pii_gen_train(self):
        
        # generate the all possible PII implemented in the create_fake_profile methods
        self._fake_labels = list(self.fake_profiles.keys())
        
        # generate the None labels 
        self._none_pii_labels = ["None" for _ in range(self._n_text)]
        
        # generate the pii labels
        self.pii_labels = sorted(self._fake_labels*self._n_text)
        
        # generate the test with no pii
        self._fake_text_no_pii = [self.faker.paragraph() for _ in range(self._n_text)]
        
        # mutiply the no pii text with the number of PII types
        self._init_fake_text_no_pii = self._fake_text_no_pii*len(self._fake_labels)     
        
        # initialize the text mixed with PII with all "None" strings. 
        self.pii_with_text = ["None" for _ in range(len(self._fake_labels)*(self._n_text))]
        
        # initialize the PII with all "None" strings
        self.PII = ["None" for _ in range(self._n_text*(len(self._fake_labels)+1))]
        
    def _random_pii_insert(self):
        # randomly insert PII into the text
        for index, PII in enumerate(tqdm(self.pii_labels)):
            # choose a PII value from the dictionary according to the PII type.
            PII_value = choose(self.fake_profiles[PII])
            
            original_fake_text = self._init_fake_text_no_pii[index]
            
            tokenized_fake_text = original_fake_text.split(" ")
            
            # generate the position to fill in the PII value
            PII_position = choose(range(len(tokenized_fake_text)+1))
            
            tokenized_fake_text.insert(PII_position, PII_value)
            
            one_text_mixed_with_PII = " ".join(tokenized_fake_text)
            
            self.pii_with_text[index] = one_text_mixed_with_PII
            self.PII[index] = PII_value
        
    
    def create_pii_text_train(self, n_text = 10):
        """
        A method to create the training text randomly mixed with fake PII. This
        method creates a text and mixed it with different kinds of PII, which leads
        to a total number of (num_of_PII)*(n_text) rows.
        
        """
        warning_text = "Please create fake profiles first with .create_fake_profile method."
        assert self.fake_profiles is not None, warning_text
        
        self._n_text = n_text
        # initialized a few variables for inserting PII values 
        self._init_pii_gen_train()
        
        # randomly insert Pii text into the paragraph.
        self._random_pii_insert()
        
        
        self.pii_with_text.extend(self._fake_text_no_pii)
        self.pii_labels.extend(self._none_pii_labels)
        
   
        
        return self.pii_labels, self.pii_with_text, self.PII
        
    def _init_pii_gen_test(self):
        
        # generate the all possible PII implemented in the create_fake_profile methods
        self._fake_labels = list(self.fake_profiles.keys())
        
        # generate the None labels 
        self._none_pii_labels = ["None" for _ in range(self._n_text)]
        
        # generate the pii labels
        self.pii_labels = sorted(self._fake_labels*self._n_text)  
        
        total_num_pii_text = (1+len(self._fake_labels))*(self._n_text)
        # generate the fake text with no pii
        self._init_fake_text_no_pii = [self.faker.paragraph() for _ in range(total_num_pii_text)]
        
        # initialize the text mixed with PII with all "None" strings. 
        self.pii_with_text = self._init_fake_text_no_pii
        # initialize the PII with all "None" strings
        self.PII = ["None" for _ in range(total_num_pii_text)]
        
    
    def create_pii_text_test(self, n_text = 10):
        """
        A method to create the testing text randomly mixed with fake PII. 
        This method creates a text and mixed it with a type of PII. 
        
        In the training text, a normal text is repeated used to insert different PIIs into
        it. In the testing text, a normal text is not intentionally repeated to insert 
        different PIIs. 
        
        """
        warning_text = "Please create fake profiles first with .create_fake_profile method."
        assert self.fake_profiles is not None, warning_text
        
        self._n_text = n_text
        
        # initialized a few variables for inserting PII values 
        self._init_pii_gen_test()
        
        # randomly insert Pii text into the paragraph.
        self._random_pii_insert()
        # add the none labels 
        self.pii_labels.extend(self._none_pii_labels)
       
        return self.pii_labels, self.pii_with_text, self.PII
        

        
def write_to_disk_train(requested_train_data_size):
    
    fake_ = Fake_PII()
    fake_.create_fake_profile(requested_train_data_size)
    train_labels, train_text, train_PII = fake_.create_pii_text_train(n_text = requested_train_data_size)
    
    # save training data to disk
    train_text_with_pii = pd.DataFrame({"Text":train_text, "Labels":train_labels, "PII":train_PII})
    train_file_name = "train_text_with_pii_" + convert_datetime_underscore(datetime.now()) + ".csv"
    train_text_with_pii.to_csv(train_file_name,index=False)

def write_to_disk_test(requested_test_data_size):
    fake_ = Fake_PII()
    fake_.create_fake_profile(requested_test_data_size)
    test_labels, test_text, test_PII = fake_.create_pii_text_test(n_text = requested_test_data_size)
    
    # save testing data to disk
    test_text_with_pii = pd.DataFrame({"Text":test_text, "Labels":test_labels, "PII":test_PII})
    test_file_name = "test_text_with_pii_" + convert_datetime_underscore(datetime.now()) + ".csv"
    test_text_with_pii.to_csv(test_file_name,index=False)
    
    
    
    
if __name__ == "__main__":
    argument_list = sys.argv[1:]
    
    if "-train" in argument_list:
        train_value_index = argument_list.index("-train") + 1
        requested_train_data_size = int(argument_list[train_value_index])
        
        write_to_disk_train(requested_train_data_size)
        
    if "-test" in argument_list:
        test_value_index = argument_list.index("-test") + 1
        requested_test_data_size = int(argument_list[test_value_index])
        
        write_to_disk_test(requested_test_data_size)
    
    
# example command line use

# python fakePII.py -train 1000 -test 100
# python fakePII.py  -test 100
# python fakePII.py -train 1000 
       
    
    
    
    
    
    
    
    
