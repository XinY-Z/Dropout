from SafeUT_dropout.DataLoader import DataLoader
import pandas as pd
from SafeUT_dropout.Vectorizers.Preprocessor import Preprocessor
from SafeUT_dropout.Vectorizers.TfIdf import Vectorizer
from SafeUT_dropout.Learners.SVM import SVM

## set hyperparameters to tune the model
alpha_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
alpha_list1 = [1e-6, 5e-7, 1e-7, 5e-8, 1e-8]
alpha_list2 = [4e-7, 2e-7, 1e-7, 8e-8, 6e-8]
alpha_list3 = [1.75e-7, 1.25e-7, 1e-7, 9e-8, 8.5e-8]

'''## load the dataset
dataloader = DataLoader(r'C:\Users\XinZ\Box\SafeUT_Data\Final_Anonymization\FINAL_ANONYMIZED_SAFEUT.xlsx', 
    sheet=['Message 1', 'Message 2', 'Message 3'])
# dataloader.plot()
dataloader.to_engagement(5)
print('passed 1')'''

file_path = r'C:\Users\XinZ\Box\SafeUT_Data\Final_Anonymization\engagement\new_engagement_october_2022'
train = pd.read_csv(file_path + r'\train_engagement_oct_24_client_talkturns.csv')
dev = pd.read_csv(file_path + r'\dev_engagement_oct_24_client_talkturns.csv')
test = pd.read_csv(file_path + r'\test_engagement_oct_24_client_talkturns.csv')

'''train = pd.read_csv('../../Downloads/SafeUT/ghosting_inspection/train_ghosting_v2.csv')
dev = pd.read_csv('../../Downloads/SafeUT/ghosting_inspection/dev_ghosting_v2.csv')
test = pd.read_csv('../../Downloads/SafeUT/ghosting_inspection/test_ghosting_v2.csv')'''


## preprocess data and convert to numeric vectors
preprocessor = Preprocessor(lowercase=True, lemma=True, remove_punc=True, remove_stopwords=False)
vectorizer = Vectorizer(ngram=(1, 2), nmessage=3, preprocessor=preprocessor)
# vectorizer.load(dataloader.data)
vectorizer.load(train, dev)
vectorizer.text2vec()
print('passed 2')

## train the model with vectorized data and tune the hyperparameter
for alpha in alpha_list1:
    print(f'\nNow using alpha={alpha}')
    svm = SVM(alpha=alpha)
    svm.kfold(n_splits=10)
    svm.evaluate(vectorizer.train_X, vectorizer.train_y, vectorizer.test_X, vectorizer.test_y)
print('all passed')

## test with the trained model
vectorizer_test = Vectorizer(ngram=(1, 2), nmessage=3, preprocessor=preprocessor)
vectorizer_test.load(train, test)
vectorizer_test.text2vec()
print('passed 2')

svm_test = SVM(alpha=9e-8)
svm_test.evaluate(vectorizer_test.train_X, vectorizer_test.train_y, vectorizer_test.test_X, vectorizer_test.test_y)
print('all passed')
