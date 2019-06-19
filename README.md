# NLP
This is a project on NLP (Sentiment Analysis) wherein I explore different models for NLP and then build my own model. 

NLPTextBlob: This file contains the python code that runs the TextBlob model on the given test for Sentiment Analysis.

2000.cvs: This file contains 25,000 entries wherein each entry has 2 subentries. TextPolarity and TextSentiment. This data has been created by running
          TextBlob moodel on 25,000 text files
          
NLTextBlobClassifier: This is the python file that runs on data from file '2000.csv'. It uses 3 different classifiers i.e. Bayes, Kernel SVM and RandomForest, to classify
                      the text into positive(1) and negative(0) based on the TextPolarity and TextSentiment values. The best accuracy I could get using these classifier
                      was 76.4%

NLPParallelDots: This is the python file that runs sentiment analysis on the ParallelDots library. I have used my own API, which will only be active
                  for the next 30 days. (See commented section in the code)
                
NLP_MyModel: This is the python file that contains the model that I built(BagOfWords Model). The model has been trained on 50,000 entries. The trained and tested
              model object has also been attached to save time while execution(refer code for loading info). Any text that needs to be sent to this model first needs to 
              preproccessed, the code for which can be found in this file itself. The bagOfowrds matrix's vocabulary has also been saved by the name 'CV_vocabulary.pkl'. 
              The data to be tested just has to be fit to this object. (Please refer the code)
              I have used 2 classifiers Bayes and RandomForest for further classifying the text into positive(1) and negative(0). I got an accuracy of
              84% on this model. (Please keep in mind that this model is very genre specific, i.e. it gives excellent results because it has been trained
              and tested on the same genre of files i.e. MovieReviews)
