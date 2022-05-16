# %%
from __future__ import annotations
from unicodedata import numeric
import numpy as np 
from pathlib import Path
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import torch
from transformers import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertTokenizer, BertModel
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from itertools import product
#import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import modAL
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import random
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pickle
import json
import codecs
import openpyxl
import ast
import torch
from torch import Tensor
import re
from sklearn.metrics import precision_recall_curve
# %%
random_state = 0
# %%
#   1. Creating a data class
#   Purpose: Transforming the database into a data object.
class Data:
    def __init__(self):
        self.data_path = Path("data")
        self.db_name = "prod.db"

    def read_data(self): # Loading data from prod.db and UN_annotation_data.csv
        con = sqlite3.connect(self.data_path/self.db_name)
        
        #self.df_annotated = pd.read_sql("SELECT * FROM annotations WHERE verifikation == 'KORREKT'", con) # Already annotated data
        self.df_prod = pd.read_sql("SELECT * FROM annotations", con) # Prod file
        self.df_UN = pd.read_csv("UN_annotation_data.csv") # The full UN dataset

        # Save label distribution
        self.df_prod["korrekt_annotering"] = self.df_prod["korrekt_annotering"].replace({"MIXED-(revisit)":"MIXED"})
        pickle.dump(self.df_prod["korrekt_annotering"][self.df_prod["korrekt_annotering"].notna()], open("active_learning_iterations/Iteration 0/BERT-RF_labeldist_iteration0","wb"))
        self.df_prod["korrekt_annotering"] = self.df_prod["korrekt_annotering"].replace({"MIXED":"MIXED-(revisit)"})

    def visualize_data_hist(self):
        assert hasattr(self, "df_prod"), "must load data first"
        self.histogram = self.df_prod.notna()["korrekt_annotering"].hist()
        #plt.savefig("test.png")

    def concat_prod(self, df_list):
        i = 1
        self.df_concat_prod = self.df_prod.copy()
        for df in df_list:
            self.df_concat_prod = pd.concat([self.df_concat_prod, df])
            #pickle.dump(df["korrekt_annotering"], open("active_learning_iterations/Iteration " + str(i) + "/BERT-RF_labeldist" + "_iteration" + str(i),"wb"))
            i += 1
    
    def preprocessing(self):
        assert hasattr(self, "df_prod"), "must load data first"
        # Preprocessing the original UN dataset
        self.df_UN["UN_idx"] = self.df_UN.index # Add permanent index tracker
        self.df_UN.dropna(inplace=True) # Dropped 1943 nulls in paragraph_text, nothing else
        self.df_UN = self.df_UN[self.df_UN["paragraph_text"].str.len() > 30] # Dropped 13543 paragraphs from being too short

        # Preprocessing prod data, removing mixed and those with paragraph lengths less than 30
        self.df_prod = self.df_prod[["index", "paragraph_text", "korrekt_annotering"]]
        self.df_prod = self.df_prod[self.df_prod["korrekt_annotering"] != "MIXED-(revisit)"]
        self.df_prod = self.df_prod[self.df_prod["paragraph_text"].map(len) > 30]
        self.df_prod.rename(columns={"index":"UN_idx"}, inplace=True) # Rename index to UN_idx to be consistent, only needed for the first iteration
        self.df_prod["embeddings"] = None
        self.df_prod['embeddings'] = self.df_prod['embeddings'].astype('object')

    def get_sample(self):
        self.df_UN_sample = self.df_UN.sample(50000, random_state=random_state) # Sample 50,000 from UN dataset
        self.df_UN_sample = self.df_UN_sample[["UN_idx", "paragraph_text"]] # Creating a df with [UN_idx, paragraph_text, korrekt_annotering, embedding]
        self.df_UN_sample["korrekt_annotering"] = None
        self.df_UN_sample["embeddings"] = None
        self.df_UN_sample['embeddings'] = self.df_prod['embeddings'].astype('object')
        # Remove examples which have already been annotated
        data.df_UN_sample = data.df_UN_sample[data.df_UN_sample["UN_idx"].isin(data.df_prod["UN_idx"][data.df_prod["korrekt_annotering"].notna()]) == False]

    def data_encoding(self):
        self.df_prod["korrekt_annotering"] = self.df_prod["korrekt_annotering"].replace(["NOT-RELEVANT","FACTUAL","SOCIAL","TEMPORAL"],[0,1,2,3])

    """
    def concat_prods(self):
        # change self.df_prod to the concatenated prods with true annotations. Run this before getting UN sample.
    """
# %%
#   2. Creating Embedding class. 
class Embedding(Data):
    def __init__(self, df, tokenizer):
        self.embeddings_list = []
        self.failed_df = pd.DataFrame()
        self.df = df # Expecting a df with [UN_idx, paragraph_text, korrekt_annotering, embedding]
        self.tokenizer = tokenizer
        super().__init__()

    def load_model(self, model):
        # Load pre-trained model (model weights)
        self.model = model
        # Put the model in evaluation mode (feed-forward operation)
        try:
            self.model.eval()
        except:
            print("Could not put model in evaluation mode")
    
    def text_embedding(self, text):
        assert hasattr(self, "model"), "must load model first"
        # Tokenization
        marked_text = "[CLS] " + text + " [SEP]" # NB: Need to change this so that SEP actually comes in between sentences
        tokenized_text = self.tokenizer.tokenize(marked_text) # Split the sentence into tokens
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text) # Map the token strings to their vocabulary indeces
        segments_ids = [1] * len(tokenized_text) # Mark each of the tokens as belonging to sentence "1"

        # Convert inputs to tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Run BERT, collect hidden states
        with torch.no_grad():

            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
        
        # For the 5th token in our sentence, select its feature values from layer 5.
        token_i = 5
        layer_i = 5
        batch_i = 0
        vec = hidden_states[layer_i][batch_i][token_i]

        # Concatenate the tensors for all layers. Stack creates new dimension.
        token_embeddings = torch.stack(hidden_states, dim=0)
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # Creating word vectors, `token_embeddings` is a [22 x 12 x 768] tensor.
        token_vecs_cat = [] # Stores the token vectors, with shape [22 x 3,072]

        # For each token in the sentence...
        for token in token_embeddings: # `token` is a [12 x 768] tensor
            # Concatenate the vectors (that is, append them together) from the last four layers.
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0) # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            token_vecs_cat.append(cat_vec) # Use `cat_vec` to represent `token`.
        
        # Sentence vectors
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
        return sentence_embedding
    
    def get_embeddings_from_row(self): # For each paragraph, run text_embedding to retrive the embeddings
        assert hasattr(self, "model"), "must load model first"
        for idx, row in self.df.iterrows():
            try:
                embedded_text = self.text_embedding(row[1]) # Expecting a df with [UN_idx, paragraph_text, korrekt_annotering, embedding]
                self.df.at[idx, 'embeddings'] = embedded_text #.cpu().detach().numpy()
            except:
                self.failed_df.append(row)
                print("Failed at: ", row[0])
    
    def get_X(self): # Turn the list of torch embeddings into a list of numpy arrays
        #assert self.embeddings_list, "Must first run get_embeddings_from_row"
        embeddings_array_list = []
        for x in self.df["embeddings"]:
            embeddings_array_list.append(x.cpu().detach().numpy())
        #self.embeddings_array_list = embeddings_array_list
        return embeddings_array_list
# %%
class Classification(Embedding):
    def __init__(self):
        pass
        #super().__init__()

    def load_classifier(self, classifier):
        self.clf = classifier
    
    def split_dataset(self, X, y):
        #train_ratio = 0.75
        #validation_ratio = 0.15
        #test_ratio = 0.10 # Drop validation set, create a 20% test set.
        self.X = X
        self.y = y

        # Shuffeling the data
        """
        c = list(zip(self.X, self.y))
        random.Random(0).shuffle(c)
        self.X, self.y = zip(*c)
        """
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=1 - train_ratio, shuffle=False)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_test, self.y_test, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=False) 
        """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)#, random_state=random_state) # Random state zero yields too good results, overfitted?

    def fitting(self):
        self.clf.fit(self.X_train, self.y_train)

    def classify(self):
        self.y_pred = self.clf.predict(self.X_test)

    def get_proba(self, pred_prob_X):
        self.pred_prob = self.clf.predict_proba(pred_prob_X) # Have to rename this to not have the same name as the function

    def cross_val(self):
        scoring = ["f1_macro", "f1_micro", "f1_weighted", "accuracy"]
        #scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_micro', 'roc_auc','precision_micro']
        kfold = KFold(n_splits=10, random_state=0, shuffle=True)
    
        self.scores = cross_validate(self.clf, self.X, self.y, scoring=scoring, cv=kfold, return_estimator=True, error_score="raise") 
        sorted(self.scores.keys())
        #for key in self.scores:
            #print(key, self.scores[key])
    
    def grid_search(self, params):
        self.params = params
        self.pipeline = make_pipeline(GridSearchCV(RandomForestClassifier(n_jobs=-1, random_state=random_state), params, scoring="f1_micro", cv=10, verbose=10))

        self.pipeline.fit(self.X_train, self.y_train) # Change to Val
        self.pipeline_y_pred = self.pipeline.predict(self.X_val)

    def randomized_grid_search(self, r_params):
        self.r_params = r_params
        self.r_pipeline = make_pipeline(RandomizedSearchCV(RandomForestClassifier(n_jobs=-1, random_state=random_state), r_params, scoring="f1_micro", cv=10, n_iter=50, random_state=random_state, verbose=10))

        self.r_pipeline.fit(self.X_train, self.y_train)
        #self.r_pipeline_y_pred = self.r_pipeline.predict(self.X_val)
# %%
class Visualize():
    def __init__(self):
        self.colors = sns.color_palette("deep")

    def plot_results(self, y_test, y_pred):
        target_names = ["N", "F","S", "T"]
        cr = classification_report(y_test, y_pred, target_names = target_names)
        #cm = confusion_matrix(y_test, y_pred)
        print(cr)

        sns_cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
        sns.heatmap(sns_cm, annot=True, xticklabels=target_names, yticklabels=target_names)
        plt.show()
    
    def plot_label_dist(self, df_list):
        for df in df_list:
            df.hist()
    
    def precision_rec(self, clf, X_test, y_test, y_score):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
        from sklearn.model_selection import train_test_split as tts
        from yellowbrick.classifier import PrecisionRecallCurve

        # Load dataset and encode categorical variables
        X, y = clf_bert_rf.X, clf_bert_rf.y
        #X = OrdinalEncoder().fit_transform(X)
        #y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, shuffle=True, random_state=0)

        # Create the visualizer, fit, score, and show it
        viz = PrecisionRecallCurve(
            RandomForestClassifier(n_estimators=2900, min_samples_leaf=1, max_features='auto', max_depth=39, criterion='entropy', bootstrap=False, random_state=0),
            per_class=True,
            cmap="Set1"
        )
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.show()

    def plot_roc_curve(self, clf, X_test, y_test, y_score):
        y_test_bin = label_binarize(y_test, classes=[0,1,2,3])
        n_classes = y_test_bin.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        list_of_auc = []

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            plt.plot(fpr[i], tpr[i], color=self.colors[i], lw=2)
            print('AUC for Class {}: {}'.format(i+1, auc(fpr[i], tpr[i])))
            list_of_auc.append(round(auc(fpr[i], tpr[i]), 2))

        print(list_of_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curves')
        legend_drawn_flag = True
        plt.text(0.58, 0.35, 'Average AUC = ', fontsize = 10)
        plt.legend(["Not-Relevant (AUC = 0.82)", "Factual (AUC = 0.73)", "Social (AUC = 0.70)", "Temporal (AUC = 0.70)"], loc=0, frameon=legend_drawn_flag)    
        plt.savefig('active_learning_iterations/Figures/ROC-AUC.png', dpi=300)
        plt.show()
        
    def gather_results(self):
        self.metrics_dict = {"classreport":[], "confusionmatrix":[], "crossval":[], "model":[], "labeldist":[]}

        for met in self.metrics_dict:
            for i in range(5):
                self.metrics_dict[met].append(pickle.load(open("active_learning_iterations/Iteration " + str(i) + "/BERT-RF_" + str(met) + "_iteration" + str(i), "rb")))
    
    def gather_aggregated(self):
        self.metrics_dict_agg = {"classreport":[], "confusionmatrix":[], "crossval":[], "model":[]}

        for met in self.metrics_dict_agg:
            for i in range(5):
                self.metrics_dict_agg[met].append(pickle.load(open("active_learning_iterations/test/BERT-RF_" + str(met) + "_iteration" + str(i), "rb")))
                #self.metrics_dict_agg[met].append(pickle.load(open("active_learning_iterations/Aggregated/" + str(i) + "_BERT-RF_" + str(met) + "_iteration" + str(i), "rb")))
# %%
class ActiveLearning():
    # First add probabilities to df
    # Then create prod file
    # Create new prod file from input (which is the predicted dataset.)
    def __init__(self):
        pass

    def merge_probas(self, df, arr): # Merge probabilities with dataframe
        arr = pd.DataFrame(arr)
        temp_df = df.reset_index()
        self.df_AL = temp_df.join(arr)

    def prep_sample(self): # Prepare dataframe to get lowest 100 of each class
        self.df_AL["max_prob"] = self.df_AL.iloc[:,-4:].max(axis=1) # First collect max predicted probability #self.df_AL["max_prob"] = active_learner.df_AL.iloc[:,-4:].max(axis=1) # First collect max predicted probability
        self.df_AL = self.df_AL.sort_values(by=["max_prob"]) # Sort by max predicted probability
        self.df_AL["predicted_class"] = self.df_AL.iloc[:,-5:-1].idxmax(axis=1) #self.df_AL["predicted_class"] = active_learner.df_AL.iloc[:,-5:-1].idxmax(axis=1) # Create new column with predicted class
        
    def get_sample(self):
        self.df_AL_sample = self.df_AL.copy()
        self.df_AL_sample["label_next"] = None

        for i in range(4):
            index_list = self.df_AL.index[self.df_AL["predicted_class"] == i].tolist()[0:100] # Select the index of the 100 first occurences of each class
            for idx in index_list: # For each index
                self.df_AL_sample["label_next"][self.df_AL_sample.index == idx] = 1 # Change this to get the dataframe
        
        self.df_AL_sample = self.df_AL_sample[self.df_AL_sample["label_next"] == 1]
    
    def transform_to_prod(self):
        self.df_AL_sample = self.df_AL_sample[["UN_idx", "paragraph_text","korrekt_annotering","embeddings"]]
        self.df_AL_sample["embeddings"] = None
# %%
    # Loading and Preprocessing the data
data = Data()
data.read_data()
data.preprocessing()
# For iteration 2:
# Reading new labels from iterations, adding them to df_prod
# %%
iter_number = 4
df_list = []
for i in range(iter_number+1)[1:]:
    df = pd.read_excel("active_learning_iterations/Dataset_Iterations/iteration_" + str(i) + ".xlsx")[["UN_idx", "paragraph_text", "korrekt_annotering", "embeddings"]]
    df["korrekt_annotering"] = df["korrekt_annotering"].replace({"SOCIAL ":"SOCIAL", "TEMPORAL ":"TEMPORAL", "MIXED_(revisit)":"MIXED", "MIXED-(revist)":"MIXED"})
    pickle.dump(df["korrekt_annotering"], open("active_learning_iterations/Iteration " + str(i) + "/BERT-RF_labeldist" + "_iteration" + str(i),"wb"))
    df = df[(df["korrekt_annotering"] != "MIXED-(revist)") & (df["korrekt_annotering"] != "MIXED_(revisit)") & (df["korrekt_annotering"] != "MIXED")]
    df_list.append(df)
# %%
""""
df_iteration_1 = pd.read_excel("active_learning_iterations/Iteration 1/label_next_iteration1.xlsx")[["UN_idx", "paragraph_text", "korrekt_annotering", "embeddings"]]
df_iteration_1 = df_iteration_1[df_iteration_1["korrekt_annotering"] != "MIXED-(revist)"]
df_iteration_2 = pd.read_excel("active_learning_iterations/Iteration 2/label_next_iteration2.xlsx")[["UN_idx", "paragraph_text", "korrekt_annotering", "embeddings"]]
df_iteration_2 = df_iteration_2[df_iteration_2["korrekt_annotering"] != "MIXED_(revisit)"]
df_iteration_2["korrekt_annotering"] = df_iteration_2["korrekt_annotering"].replace({"SOCIAL ":"SOCIAL", "TEMPORAL ":"TEMPORAL"})
#df_list = [df_iteration_1, df_iteration_2]
"""

data.concat_prod(df_list)
data.df_prod = data.df_concat_prod
# %%
# Saving labeling iteration 
#data.df_prod[data.df_prod["korrekt_annotering"].notna()].to_csv("active_learning_iterations/prod_annotated_iteration1.csv")
# Need to reset index
data.df_prod = data.df_prod.reset_index()[["UN_idx", "paragraph_text", "korrekt_annotering", "embeddings"]]
data.data_encoding()
data.get_sample()
# %%
    # Loading the models and tokenizer
tokenizer = tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
rf = RandomForestClassifier(random_state=random_state)
# %%
# Or load with same parameters
rf = RandomForestClassifier(n_estimators=2900, min_samples_leaf=1, max_features='auto', max_depth=39, criterion='entropy', bootstrap=False, random_state=0)
# TO-DO: Add TF-IDF. Create a load_tfidf method in the embeddings class
# %%
    # Retrieving embeddings for annotated data
###
annotated_embeddings = Embedding(data.df_prod[data.df_prod["korrekt_annotering"].notna()].copy(), tokenizer) # Retriving embeddings only from annotated data.
annotated_embeddings.load_model(bert)
annotated_embeddings.get_embeddings_from_row() 
# %%
# Saving labeling/embeddings iteration backup
#annotated_embeddings.df.to_csv("active_learning_iterations/Iteration 2/prod_annotated_embeddings_iteration2.csv")
# %%
    # Train random forest classifier on annotated data (Or load model)
    # Current mode: Load

clf_bert_rf = Classification()
clf_bert_rf.load_classifier(rf)
# %%
annotated_embeddings.df = annotated_embeddings.df.sample(frac=1, random_state=random_state) # Shuffle dataset
clf_bert_rf.split_dataset(annotated_embeddings.get_X(), annotated_embeddings.df["korrekt_annotering"])
# %%
clf_bert_rf.fitting() # Fit the model
# %%
clf_bert_rf.classify() # classify() also fits the model, in addition to predicting.
#clf_bert_rf.cross_val()
# %%
np.average(clf_bert_rf.scores["test_accuracy"]) # Print out average f1 
# %%
# %%
    # Saving model iteration, Cross Validation scores, Confusion Matrix, Classificaiton Report
"""
i = iter_number
pickle.dump(clf_bert_rf.clf, open("active_learning_iterations/Iteration " + str(i) + "/BERT-RF_model_iteration" + str(i), 'wb'))
pickle.dump(clf_bert_rf.scores, open("active_learning_iterations/Iteration " + str(i) + "/BERT-RF_crossval_iteration" + str(i), 'wb'))
#pd.DataFrame(clf_bert_rf.scores.items(), columns=["metric","values"]).to_csv("active_learning_iterations/Iteration 2/BERT-RF_crossval_scores_iteration2.csv")

target_names = ["N", "F", "S", "T"] 
pickle.dump(classification_report(clf_bert_rf.y_test, clf_bert_rf.y_pred, target_names = target_names), open("active_learning_iterations/Iteration "+ str(i) + "/BERT-RF_classreport_iteration"+ str(i), 'wb'))
pickle.dump(confusion_matrix(clf_bert_rf.y_test, clf_bert_rf.y_pred), open("active_learning_iterations/Iteration "+ str(i) + "/BERT-RF_confusionmatrix_iteration"+ str(i), 'wb'))
"""
# %%
# Alternative, save model in aggregated folder
"""
i = # Set based on iteration number
pickle.dump(clf_bert_rf.clf, open("active_learning_iterations/"+ str(i) + "_BERT-RF_model_iteration" + str(i), 'wb'))
pickle.dump(clf_bert_rf.scores, open("active_learning_iterations/"+ str(i) + "_BERT-RF_crossval_iteration" + str(i), 'wb'))
#pd.DataFrame(clf_bert_rf.scores.items(), columns=["metric","values"]).to_csv("active_learning_iterations/Iteration 2/BERT-RF_crossval_scores_iteration2.csv")

target_names = ["N", "F", "S", "T"] 
pickle.dump(classification_report(clf_bert_rf.y_test, clf_bert_rf.y_pred, target_names = target_names), open("active_learning_iterations/"+ str(i) + "_BERT-RF_classreport_iteration"+ str(i), 'wb'))
pickle.dump(confusion_matrix(clf_bert_rf.y_test, clf_bert_rf.y_pred), open("active_learning_iterations/"+ str(i) + "_BERT-RF_confusionmatrix_iteration"+ str(i), 'wb'))
"""
# %%
    # Alternative: Load model from iteration
clf_bert_rf = Classification()
#clf_bert_rf.load_classifier(pickle.load(open("active_learning_iterations/Tuning/best_estimator_random1_micro", 'rb'))) 
clf_bert_rf.load_classifier(pickle.load(open("active_learning_iterations/Tuning/best_estimator_random1_micro", 'rb')))  # 0 = 0.4234, 1 = 0.4020, 2 = 0.401, 3 is correct, 4 is correct
# %%
    # Active Learning Chapter
    # Get full embeddings
# Use classifier with pred_proba on the UN_sample, add these probs to UN_sample, then construct prod_db.
full_embeddings = Embedding(data.df_UN_sample.copy(), tokenizer)
full_embeddings.load_model(bert)
full_embeddings.get_embeddings_from_row()
# %%
    # Use classifier to predict full sample dataset for active learning
print(full_embeddings.df[full_embeddings.df["embeddings"].isna()]) # Check which paragraphs failed to get embedded
full_embeddings.df = full_embeddings.df[full_embeddings.df["embeddings"].notna()] # Remove the one row which failed
# Load full embeddings
"""
full_embeddings.df = pd.read_csv("active_learning_iterations/backup_df_AL_iteration1.csv")[["UN_idx", "paragraph_text", "korrekt_annotering", "embeddings"]]
"""
# %%
# Change string values of embeddings to tensors
full_embeddings.df["embeddings"] = full_embeddings.df["embeddings"].apply(lambda x: eval(x))
# Remove those rows which is already labeled
full_embeddings.df = full_embeddings.df[full_embeddings.df["UN_idx"].isin(annotated_embeddings.df["UN_idx"]) == False]
# %%
clf_bert_rf.get_proba(full_embeddings.get_X()) # Get predicition probabilities
# %%
    # Create active learning dataframe to be labeled in the next iteration
active_learner = ActiveLearning()
active_learner.merge_probas(full_embeddings.df, clf_bert_rf.pred_prob) # Concatenate full_embeddings and probabilities
active_learner.prep_sample() # Prep sample for getting 100 lowest predictions for each class
"""
# Save backup full_embeddings
active_learner.df_AL.to_csv("active_learning_iterations/backup_df_AL_iteration1.csv")
"""
active_learner.get_sample()
active_learner.transform_to_prod()
# %%
active_learner.df_AL_sample.to_excel("active_learning_iterations/Iteration 2/label_next_iteration2.xlsx") # Save active learning dataframe as csv
# %%
# Hyperparameter Tuning, Randomzied Search
clf_bert_rf = Classification()
clf_bert_rf.load_classifier(rf)
annotated_embeddings.df = annotated_embeddings.df.sample(frac=1, random_state=random_state) # Shuffle dataset
clf_bert_rf.split_dataset(annotated_embeddings.get_X(), annotated_embeddings.df["korrekt_annotering"])

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 10000, num = 1000)]
criterion = ["gini", "entropy"]
max_samples = np.array([int(x) for x in np.linspace(1, 10, num = 10)])/10
min_samples_leaf = np.array([int(x) for x in np.linspace(1, 20, num = 10)])
max_features = ['auto', 'sqrt', 'log2']
max_depth = [int(x) for x in np.linspace(1, 50, num = 10)]
#max_depth.append(None)
bootstrap = [True, False]

# Create the random grid
random_grid = {
    'n_estimators': n_estimators,
    'criterion': criterion,
    #'max_samples': max_samples,
    'min_samples_leaf': min_samples_leaf,
    'max_features': max_features,
    'max_depth': max_depth,
    'bootstrap': bootstrap
    }

clf_bert_rf.randomized_grid_search(random_grid)
# %%
clf_bert_rf = Classification()
clf_bert_rf.load_classifier(rf)
annotated_embeddings.df = annotated_embeddings.df.sample(frac=1, random_state=random_state) # Shuffle dataset
clf_bert_rf.split_dataset(annotated_embeddings.get_X(), annotated_embeddings.df["korrekt_annotering"])

# Grid search
param_grid = {
    'n_estimators': [2900],
    'criterion': ['entropy'],
    #'max_samples': [None, 100, 500],
    'min_samples_leaf': [1],
    'max_features': ['auto'],
    'max_depth': [39],
    'bootstrap': [False]
    }

clf_bert_rf.grid_search(param_grid)
# %%
clf_bert_rf.pipeline[0].best_score_
#clf_bert_rf.pipeline[0].best_params_
    # %%
pickle.dump(clf_bert_rf.r_pipeline[0].best_params_, open("best_params_grid1_micro", "wb"))
pickle.dump(clf_bert_rf.r_pipeline[0].best_score_, open("best_score_grid1_micro", "wb"))
pickle.dump(clf_bert_rf.r_pipeline[0].best_estimator_, open("best_estimator_grid1_micro", "wb"))
#clf_bert_rf.r_pipeline[0].best_score_
# %%
pickle.load(open("best_params_random1_micro", "rb"))
pickle.load(open("best_score_random1_micro", "rb"))
# %%
# Results Chapter
visualizer = Visualize()
visualizer.gather_results()
visualizer.gather_aggregated()
# %%

# %%
# Plot label distributions
values = []
labels = ['NOT-RELEVANT', 'FACTUAL', 'SOCIAL', 'TEMPORAL', 'MIXED']

for i in range(5):
    val_list = []
    for lab in labels:
        val_list.append(visualizer.metrics_dict["labeldist"][i].value_counts()[lab])
    values.append(val_list)
values.append(np.sum(values, axis=0))
# %%
dpi = 300

font_size_1 = 20
figsize_1 = (15, 10)

font_size_2 = 14
figsize_2 = (20,10)

font_size_4 = 10
figsize_4 = (15,13)
# %%
sum(values[0])
# %%
# Plot first initial iteration
fig, axs = plt.subplots(figsize=figsize_1, sharey=True)
axs.bar(labels, values[0])
plt.rc("font", size=font_size_1)
plt.title("0. Labeling Iteration")
#plt.savefig('active_learning_iterations/Figures/labeldist_iteration_0.png', dpi=dpi)
plt.show()
# %%
# Plot first and aggregated distribution
fig, axs = plt.subplots(1,2, figsize=figsize_2, sharey=True)
axs[0].bar(labels, values[0])
axs[0].title.set_text('O. Labeling Iteration')
axs[1].bar(labels, values[5])
axs[1].title.set_text('Final Aggregated Dataset')
plt.rc("font", size=font_size_2)
#plt.savefig('active_learning_iterations/Figures/labeldist_iteration_0_aggregated.png', dpi=dpi)
plt.show()
# %%
# Plot the 4 active learning iteration distribution
fig, axs = plt.subplots(2,2, figsize=figsize_4, sharey=True)
axs[0,0].bar(labels, values[1])
axs[0,0].title.set_text('1. Labeling Iteration')
axs[0,1].bar(labels, values[2])
axs[0,1].title.set_text('2. Labeling Iteration')
axs[1,0].bar(labels, values[3])
axs[1,0].title.set_text('3. Labeling Iteration')
axs[1,1].bar(labels, values[4])
axs[1,1].title.set_text('4. Labeling Iteration')
plt.rc("font", size=font_size_4)
#plt.savefig('active_learning_iterations/Figures/labeldist_iteration_1-4.png', dpi=dpi)
plt.show()
# %%
# Plot performance over iterations (non-aggregated)
f1_macro_list = [np.average(x["test_f1_macro"]) for x in visualizer.metrics_dict["crossval"][:]]
accuracy_list = [np.average(x["test_accuracy"]) for x in visualizer.metrics_dict["crossval"][:]]

fig, axs = plt.subplots(figsize=figsize_1)
plt.rc("font", size=font_size_1)
plt.title("Performance Change over Iterations")
plt.xlabel("Number of Iterations", fontsize=15)
axs.plot(accuracy_list, marker="o", label="Accuracy")
axs.plot(f1_macro_list, marker="o", label="Macro F1")
plt.legend()
axs.set_ylim([0.35, 0.65])
plt.xticks(np.arange(0, 5, step=1))
#plt.savefig('active_learning_iterations/Figures/AL_iteration_performance.png', dpi=dpi)
plt.show()
# %%
# Plot performance over iterations (aggregated dataset)
f1_macro_list = [np.average(x["test_f1_macro"]) for x in visualizer.metrics_dict_agg["crossval"][:]]
accuracy_list = [np.average(x["test_accuracy"]) for x in visualizer.metrics_dict_agg["crossval"][:]]

fig, axs = plt.subplots(figsize=figsize_1)
plt.rc("font", size=font_size_1)
plt.title("Model Performance on the Aggregated Dataset")
plt.xlabel("Model Number", fontsize=15)
axs.plot(accuracy_list, marker="o", label="Accuracy")
axs.plot(f1_macro_list, marker="o", label="Macro F1")
plt.legend()
axs.set_ylim([0.35, 0.65])
plt.xticks(np.arange(0, 5, step=1))
plt.savefig('active_learning_iterations/Figures/AL_aggergated_performance.png', dpi=dpi)
plt.show()
# %%
[np.average(x["test_f1_macro"]) for x in visualizer.metrics_dict_agg["crossval"][:]]
# %%
[np.average(x["test_f1_micro"]) for x in visualizer.metrics_dict["crossval"][:]]
# %%
visualizer = Visualize()
visualizer.plot_results(clf_bert_rf.y_test, clf_bert_rf.y_pred)
X_test = clf_bert_rf.X_test
pred_prob = clf_bert_rf.clf.predict_proba(X_test)
visualizer.plot_roc_curve(clf_bert_rf.clf, clf_bert_rf.X_test, clf_bert_rf.y_test, pred_prob)
# %%
visualizer = Visualize()
visualizer.precision_rec(clf_bert_rf.clf, clf_bert_rf.X_test, clf_bert_rf.y_test, pred_prob)
# %%
    # Investigate joining Steffen and Harvard Dataset
# Loading both datasets
df_har_speeches = pd.read_csv("data/UN_meta_speeches.csv") # Information about each speech and the speaker, 82K in total. (A speach is not a paragraph)
df_har_meetings = pd.read_csv("data/UN_meta_meetings.csv") # Information about each meeting, 5.7K in total.
# %%
df_har_meetings
# %%
df_edges = pd.read_csv("data/edges_1.csv")
df_episodes = pd.read_csv("data/episodes_1.csv")
# %%
df_episodes["resolution"].nunique()
# %%
# Renaming id column in episodes to match id in df_har_meetings
df_episodes = df_episodes.rename(columns={"meeting.record":"meeting_id"})
df_har_meetings = df_har_meetings.rename(columns={"basename":"meeting_id"})
# %%
# Changing meeting_ids to match
df_episodes['meeting_id'].apply(lambda x: x[-8:]).apply(lambda x: len(x)).unique()
# %%
df_har_meetings['meeting_id'] = df_har_meetings['meeting_id'].apply(lambda x: x[-8:]) 
df_episodes["meeting_id"] = df_episodes["meeting_id"].apply(lambda x: x[0] + x[2:]) # Remove / from meeting_id
# %%
# Add topic, etc. from df_har_meetings to df_merged by merging on meeting_id
df_merged = df_episodes.merge(df_har_meetings[["meeting_id", "num_speeches", "topic", "year", "month", "day"]], how="left", on="meeting_id").drop(columns=["Unnamed: 0.1", "Unnamed: 0"])
# %%
# Group by topic
df_group = df_merged.groupby(by="topic_y")
# %%
df_har_meetings[["topic"]].value_counts()[100:].hist()
# %%
# %%
df_har_meetings[["topic"]].value_counts().describe()
# %%
df_har_meetings[["topic"]].value_counts()[0:30].describe()
# %%
df = pd.read_csv("active_learning_iterations/Iteration 1/backup_df_AL_iteration1.csv")
# %%
# Get topic names (group name) of those groups containing a vote, sample 100 random groups, put group name into list group_keys.
#print("Amount of resolutions voted on by sampled group: ", df_group.count()[(df_group.count()["resolution"] != 1) & (df_group.count()["resolution"] <= 10)].sample(50, random_state=1)[["vote"]].sum()) 
group_keys = df_group.count()[(df_group.count()["resolution"] >= 1) & (df_group.count()["resolution"] <= 5)].sample(50, random_state=1) # NB: Add notna. #group_keys = df_group.count()[(df_group.count()["resolution"] != 1) & (df_group.count()["resolution"] <= 10)].sample(50, random_state=1) # NB: Add notna. # Chcek this one to see how large the original sample was.
print("Amount of resolutions voted on by sampled group: ", group_keys[["vote"]].sum()) 
group_keys = group_keys.index.tolist()
# %%
# Use group_keys to concatenate the 100 sampled groups
aggregated_df = pd.DataFrame()
for key in group_keys:
    aggregated_df = pd.concat([aggregated_df, df_group.get_group(key)])
# %%
# Merge UN paragraphs on aggregated_df based on doc_id
df_un_copy = data.df_UN.copy()
df_un_copy["doc_id"] = df_un_copy["doc_id"].apply(lambda x: x[10:-12])
df_un_copy = df_un_copy.rename(columns={"doc_id":"meeting_id"})
# %%
agg_merged = aggregated_df.merge(df_un_copy, how="left", on="meeting_id")
# %%
print("Amount of unique meetings: ", agg_merged["meeting_id"].nunique())
print("Amount of unique resolutions: ", agg_merged["resolution"].nunique())
print("Amount of paragraphs: ", len(agg_merged))
print("Unique topics:", agg_merged["topic_y"].nunique())
# %%
agg_merged["paragraph_text"].nunique()
# %%
# Altering agg_merged to look like df_UN_sample
agg_merged_altered = agg_merged[["UN_idx","paragraph_text"]]
agg_merged_altered["korrekt_annotering"] = None
agg_merged_altered["embeddings"] = None
# %%
agg_merged_altered_embeddings = Embedding(agg_merged_altered.copy(), tokenizer)
agg_merged_altered_embeddings.load_model(bert)
agg_merged_altered_embeddings.get_embeddings_from_row()
# %%
# Saving agg_merged_altered_df
pickle.dump(agg_merged_altered_embeddings.df, open("agg_merged_altered_embeddings_df_2", "wb"))
# %%

# %%
print(len(agg_merged_altered_embeddings.df[agg_merged_altered_embeddings.df["embeddings"].isna()])) # Check which paragraphs failed to get embedded (53 failed)

agg_merged_altered_embeddings.df = agg_merged_altered_embeddings.df[agg_merged_altered_embeddings.df["embeddings"].notna()] # Remove the one row which failed
# %%
# Get probabilities for each paragraph in X
clf_bert_rf.get_proba(agg_merged_altered_embeddings.get_X())
# %%
active_learner_agg = ActiveLearning()
active_learner_agg.merge_probas(agg_merged_altered_embeddings.df, clf_bert_rf.pred_prob)
active_learner_agg.prep_sample()
# %%
#pickle.dump(agg_merged_altered_embeddings.df, open("agg_merged_altered_embeddings_df_stage2", "wb"))
# %%
# Add UN_idx to df_AL
al_docid_merge = active_learner_agg.df_AL.merge(data.df_UN[["doc_id", "UN_idx"]], how="left", on="UN_idx")
#pickle.dump(agg_merged_altered_embeddings.df, open("agg_merged_altered_embeddings_df_stage3", "wb"))

# %%
# Merge agg_merged with deparadoxification strategies (al_docid_merge)
df_votes_strategies = agg_merged.merge(al_docid_merge, how="left", on="UN_idx")
#pickle.dump(df_votes_strategies, open("agg_merged_altered_embeddings_df_stage4", "wb"))
# %%
#df_votes_strategies = pickle.load(open("active_learning_iterations/Exploring Deparadoxification/agg_merged_altered_embeddings_df_stage4", "rb"))
# %%
# Removing irrelevant columns, incl. embeddings
df_votes_strat_strip = df_votes_strategies.copy()
df_votes_strat_strip = df_votes_strat_strip.drop(columns=["doc_id", "embeddings", "text", "index", "korrekt_annotering", "topic_x", 0, 1, 2, 3, "paragraph_text_x", "Unnamed: 0", "duration", "year", "month", "day"])
# %%
df_votes_strat_strip = df_votes_strat_strip.sort_values(by=["topic_y", "date", "meeting_id", "paragraph_number"])
df_votes_strat_strip["sum_NN"] = ""
df_votes_strat_strip["sum_F"] = ""
df_votes_strat_strip["sum_S"] = ""
df_votes_strat_strip["sum_T"] = ""
# %%
df_votes_strat_strip["drop"] = 0
# %%
# Sum all above until you hit a vote, unless the meeting id/res id is the same, then continue.
df_vss_group = df_votes_strat_strip.groupby(by=["topic_y"])
# %%
list_of_dfs = []
# %%
# Add relevant groups to list of dataframes
i = 0
for name, group in df_vss_group:
    #df_vss_group.get_group(list(df_vss_group.groups.keys())[i])
    if group["resolution"].notna().sum() == 0:
        #df_vss_group.get_group(list(df_vss_group.groups.keys())[i])["drop"] = 1
        i += 1
    else:
        df = df_vss_group.get_group(name)
        list_of_dfs.append(df_vss_group.get_group(name))
print(i)
# %%
# Change df to a loop going through each group
new_list = []
for df in list_of_dfs:
    #df = list_of_dfs[6] #df = df_vss_group.get_group(list(df_vss_group.groups.keys())[26]).copy()
    df["last_vote"] = ""
    df["res_copy"] = df["resolution"]

    df = df.reset_index()
    idx_list = []

    # Get indexes of all the last votes
    for res in df["resolution"].unique():
        s = df["resolution"]
        idx = s.where(s==res).last_valid_index()
        idx_list.append(idx)

    # Add last_vote to all the last votes
    for idx in enumerate(idx_list):
        try:
            df.loc[idx[1], "last_vote"] = "last"
        except:
            pass

    idx_list = [i for i in idx_list if i]

    df["temp_idx"] = df.index # Save actual index
    first = True 

    # Go through index list of all the last voting paragraphs
    for list_idx, idx in enumerate(idx_list):
        if first == True: # Check if its the first last voting paragraph in the dataframe
            add_idx = 0
            first = False
        else:
            add_idx = idx_list[list_idx-1]+1

        # Get value counts of predicted classes
        val_count = df.loc[add_idx:idx, "predicted_class"].value_counts()
        tuples = [tuple((x, y)) for x, y in val_count.items()]

        # Add sum of classes to the dataframe
        for t in tuples:
            if t[0] == 0.0:
                df.loc[add_idx:idx, "sum_NN"] = t[1]
            elif t[0] == 1.0:
                df.loc[add_idx:idx, "sum_F"] = t[1]
            elif t[0] == 2.0:
                df.loc[add_idx:idx, "sum_S"] = t[1]
            elif t[0] == 3.0:
                df.loc[add_idx:idx, "sum_T"] = t[1]
    new_list.append(df)
# %%
concat_df = pd.concat(new_list)
# %%
#concat_df.to_excel("agg_predicted_sum_metadata.xlsx")
# %%
slim_df = concat_df[concat_df["last_vote"]=="last"][["vote","sum_NN","sum_F","sum_S","sum_T"]]
#slim_df.to_csv("slim_df.csv")
slim_df = slim_df.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace('', np.nan)
slim_df = slim_df.fillna(0)
# %%
def voting_scaling(vote):
    pro = int(re.search('(\d*).*', vote)[1])
    contra = int(re.search('\d*-(\d*).*', vote)[1])
    neutral = int(re.search('\d*-\d*-(.*)', vote)[1])
    points = (pro*2 - contra*2 + neutral) / 30
    #print(pro, contra, neutral)
    #print(vote)
    #print(points)
    #print("------")
    return points
# %%
# Dropping first row (probably not necessary)
slim_df = slim_df[(slim_df["vote"] != "without vote") & (slim_df["vote"] != "14-0-1, 15-0-0")]
# %%
slim_df["points"] = slim_df["vote"].apply(voting_scaling)
# %%
# Importing regression models
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
# %%
# Load model & dataset
#reg_clf = pickle.load(open("regression_clf_V2", "rb"))
#slim_df = pd.read_csv("slim_df_V2.csv")
# %%
y = slim_df["points"]
X = slim_df[["sum_NN", "sum_F", "sum_S", "sum_T"]]
# %%
# Fitting regression model
reg_clf = LinearRegression().fit(X, y)
# %%
# Plotting results
#[round(x, 10) for x in reg_clf.coef_]
#reg_clf.score
reg_clf.score(X, y)
# %%
import numpy as np
import statsmodels.api as sm

y = slim_df["points"]
X = slim_df[["sum_NN", "sum_F", "sum_S", "sum_T"]]

# Add constant
X = sm.add_constant(X)
# Fit and summarize OLS model
mod = sm.OLS(y, X)
res = mod.fit()

# Print results
print(res.summary())
#print(res.summary().as_latex())
#print(res.summary2().as_latex())
# %%
res.pvalues
# %%
# Experimenting with plotting the OLS
#fig = sm.graphics.influence_plot(res, criterion="cooks")
#fig.tight_layout(pad=1.0)
plt.figure()
sm.graphics.plot_partregress_grid(res)
plt.tight_layout()
plt.savefig('active_learning_iterations/Figures/partial_reg_plot.png', dpi=dpi)
plt.show()
#fig.tight_layout(pad=1.0)
# %%

# %%
# Save model and dataset
""""
pickle.dump(reg_clf, open("regression_clf_V2", "wb"))
slim_df.to_csv("slim_df_V2.csv")
"""
# %%
slim_df[["sum_NN", "sum_F", "sum_S", "sum_T", "points"]].value_counts()
slim_df[["points"]].value_counts()
# %%
dpi = 300

font_size_1 = 20
figsize_1 = (15, 10)

font_size_2 = 14
figsize_2 = (20,10)

font_size_4 = 10
figsize_4 = (15,13)

# Plot first initial iteration
fig, axs = plt.subplots(figsize=figsize_1, sharey=True)
plt.hist(slim_df[["points"]])
plt.rc("font", size=font_size_1)
plt.title("Regression Dataset Points Distribution")
plt.xlabel("Points", fontsize=20)
plt.ylabel("Count", fontsize=20)
plt.savefig('active_learning_iterations/Figures/regression_points.png', dpi=dpi)
plt.show()
# %%