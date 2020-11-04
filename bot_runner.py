# this is the bot_runner.py file for running the bot on root level
# the bot_runner is composed of two part
#   part 1: sentiment analysis
#   part 2: comment generation
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model.comment_generation import Markov
from model.sentiment_analysis import GNBModel, KMeansModel, StochasticGradientDescent, SVMModel

def read_csv_to_df():
    file = open('.config', "r")
    read = file.read()
    for line in read.splitlines():
        if 'post_filename =' in line:
            post_df = pd.read_csv("./data/"+line.split('=',1)[1][2:-1]) #['message', 'post_id', 'react_angry', 'react_haha', 'react_like', 'react_love', 'react_sad', 'react_wow', 'shares', 'label']
        elif 'comment_filename =' in line:
            comment_df = pd.read_csv("./data/"+line.split('=',1)[1][2:-1]) #['created_time', 'from_id', 'from_name', 'message', 'post_id']
        elif 'train_ratio =' in line:
            train_ratio = float(line.split('=',1)[1])
        elif 'test_ratio =' in line:
            test_ratio = float(line.split('=',1)[1])
        elif 'val_ratio =' in line:
            val_ratio = float(line.split('=',1)[1])
    return post_df, comment_df, train_ratio, test_ratio, val_ratio

def remove_null_rows(df):
    remove_indexes = []
    for index, row in df.iterrows():
        if row['react_angry'] == row['react_haha'] == row['react_like'] == row['react_love'] == row['react_sad'] == row['react_wow'] == 0:
            remove_indexes.append(index)
        elif row['message'] == np.nan:
            remove_indexes.append(index)
    return df.drop(remove_indexes)

def get_sentiment(reactions):
    max_reactions = []
    for i in range(len(reactions[0])):
        max_reaction = None
        max_reaction_count = 0
        for j in range(len(reactions)):
            if reactions[j][i] > max_reaction_count:
                max_reaction_count = reactions[j][i]
                max_reaction = j
        max_reactions.append(max_reaction)

    sentiment = [0]*len(max_reactions)
    for i in range(len(max_reactions)):
        if max_reactions[i] == 2 or max_reactions[i] == 3:
            sentiment[i] = 1
        elif max_reactions[i] == 0 or max_reactions[i] == 4:
            sentiment[i] = -1
    return sentiment

def train_test_val_split(df, train_ratio, test_ratio, val_ratio=0):
    # train test ratio should sum up to 1 if val ratio is not provided
    # otherwise, the three ratios should sum up to 1
    print('splitting train test val data ...')
    if train_ratio + test_ratio + val_ratio - 1 > 1e-5:
        print("please provide eligible train, test, val ratio")
        return []*27
    else:
        X = df['message'].to_list()
        y1 = df['react_angry'].to_list()
        y2 = df['react_haha'].to_list()
        y3 = df['react_like'].to_list()
        y4 = df['react_love'].to_list()
        y5 = df['react_sad'].to_list()
        y6 = df['react_wow'].to_list()
        y0 = get_sentiment([y1, y2, y3, y4, y5, y6])
        y7 = df['comment_list'].to_list()
        X_train, X_test, y0_train, y0_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test, \
        y4_train, y4_test, y5_train, y5_test, y6_train, y6_test, y7_train, y7_test = train_test_split(X, y0, y1, y2, y3, y4, y5, y6, y7, test_size=test_ratio, random_state=1)
        X_val = y0_val = y1_val = y2_val = y3_val = y4_val = y5_val = y6_val = y7_val = None
        if val_ratio != 0:
            X_train, X_val, y0_train, y0_val, y1_train, y1_val, y2_train, y2_val, y3_train, y3_val, \
            y4_train, y4_val, y5_train, y5_val, y6_train, y6_val, y7_train, y7_val = train_test_split(X_train, y0_train, y1_train, y2_train, y3_train, \
                y4_train, y5_train, y6_train, y7_train, test_size=val_ratio/(train_ratio+val_ratio), random_state=1) # test_size = second split ratio between train and val
        return [X_train, X_val, y0_train, y0_val, y1_train, y2_train, y3_train, y4_train, y5_train, y6_train, \
            y1_val, y2_val, y3_val, y4_val, y5_val, y6_val, X_test, y0_test, y1_test, y2_test, y3_test, y4_test, y5_test, y6_test, y7_train, y7_val, y7_test]

def generate_post_comment_nested_dict(post, comment, n):
    def ngrams(comment, n):
        output = []
        if isinstance(comment, str):
            comment = comment.split(' ')
            for i in range(len(comment)-n+1):
                output.append(comment[i:i+n])
        return output

    # merge post and comment df and only take the messsage and comment columns 
    print('generating post comment nested dict ...')
    comment = comment.rename(columns={"message": "comment", "post_id": "post_id_1"})
    new_df = pd.merge(post, comment, left_on='post_id', right_on='post_id_1', how='left')
    
    post_comment_dict = {}
    if n == 0: # get post - comment dict
        for index, row in new_df.iterrows():
            post = row['message']
            comment = row['comment']
            # construct a nested dict where the key is post 
            # and the value is a list of all comments
            comment_list = post_comment_dict.get(post, [])
            comment_list.append(comment)
            post_comment_dict[post] = comment_list
            if index % 1000 == 0:
                print('post comment pair ' + str(index) + ' added')
    else:  # perform n gram to generate word pairs for words in comments
        for index, row in new_df.iterrows():
            post = row['message']
            ngram_list = ngrams(row['comment'], n)
            # construct a nested dict where the key is post 
            # and the value is a word dict for n-gram words from its comments
            word_dict = post_comment_dict.get(post, {})
            for word_1, word_2 in ngram_list:
                if word_1 in word_dict.keys():
                    word_dict[word_1].append(word_2)
                else:
                    word_dict[word_1] = [word_2]
            post_comment_dict[post] = word_dict
            if index % 1000 == 0:
                print('post comment pair ' + str(index) + ' added')
    return post_comment_dict

def transform_dict_to_df(data_dict, post_df, word_pair_flag):
    print('tranforming post comment dict to post reaction comment dataframe ...')
    post_df['comment_list'] = np.empty((len(post_df), 0)).tolist()
    for index, row in post_df.iterrows():
        key = row['message']
        word_list = data_dict.get(key)
        if word_pair_flag is True:
            word_pairs = []
            for word_1 in word_list:
                for word_2 in word_list.get(word_1):
                    word_pair = (' ').join([word_1, word_2]) #word_pair = "word1 word2"
                    word_pairs.append(word_pair)
            post_df.at[index, 'comment_list'] = word_pairs
        else:
            post_df.at[index, 'comment_list'] = word_list
        if index % 100 == 0:
                print('post ' + str(index) + '\'s dataframe row generated')
    post_df = post_df.drop_duplicates(subset=['message']) # there are duplicate posts
    return post_df

def main():
    # if val_ratio=0, we will only use test_ratio for training, so train_ratio = 1 - test_ratio
    post_df, comment_df, train_ratio, test_ratio, val_ratio = read_csv_to_df() 
    # switch for two problems
    sentiment_analysis = True
    comment_generation = True
    X_train_cluster = []
    X_val_cluster = []

    post_comment_dict = generate_post_comment_nested_dict(post_df, comment_df, 0) # post - comment dataset
    post_comment_df = transform_dict_to_df(post_comment_dict, post_df, False)
    data_list = train_test_val_split(post_comment_df, train_ratio, test_ratio, val_ratio)

    if data_list is not None:
        # problem 1: sentiment analysis
        # reminder: the data split is different each time we run the program
        #           so the models are only comparable from the same run
        X_train = data_list[0]
        X_val = data_list[1]
        X_test = data_list[16]
        y0_train = data_list[2]
        y0_val = data_list[3]
        y0_test = data_list[17]
        
        if sentiment_analysis is True:
            # baseline: gaussian naive bayes
            gnb = GNBModel()
            model = gnb.train(X_train, y0_train)
            gnb.predict(model, X_test, y0_test, 'gnb')
            
            # method 1: sgd model
            sgd = StochasticGradientDescent()
            model = sgd.train(X_train, y0_train, X_val, y0_val, losses=['hinge', 'log', 'squared_loss'], learning_rates=['optimal'], tols=[1e-5, 1e-4, 1e-3], max_iter=500)
            sgd.predict(model, X_test, y0_test, 'sgd')

            # method 2: svm model 
            svm = SVMModel()
            model = svm.train(X_train, y0_train, X_val, y0_val, kernels=['linear', 'poly', 'rbf', 'sigmoid'], gammas=['auto', 'scale'], tols=[1e-5, 1e-4, 1e-3], max_iters=[300, 500])
            svm.predict(model, X_test, y0_test, data_list[18:24], 'svm')

        if sentiment_analysis is True or comment_generation is True:
            # method 3: kmeans model
            kmeans = KMeansModel()
            model, X_train_cluster, X_val_cluster = kmeans.train(X_train, X_val, y0_val, data_list[10:16], n_clusters_list=[6, 12, 18, 24], tols=[1e-6, 1e-5, 1e-3], max_iters=[300, 500])
            X_test_cluster = kmeans.predict(model, X_test, y0_test, data_list[18:24], 'kmeans')

        # problem 2: comment generation
        if comment_generation is True: 
            y7_train = data_list[24]
            y7_val = data_list[25]
            y7_test = data_list[26]

            # step0 kmeans: generate and save markov text dict for cluster from best kmeans model in problem 1
            # baseline: kmeans + vanilla markov chain / pos-tagging-improved markov chain
            markov = Markov(X_train, X_test, y7_train, y7_test)
            markov_text_dict = markov.generate_markov_dict(X_train, X_train_cluster, y7_train)
            markov.generate_comment(markov_text_dict, X_test, X_test_cluster, y7_test, False, 'markov_vanilla.txt')

            # method improved: kmeans + markov chain combined model
            markov.generate_comment(markov_text_dict, X_test, X_test_cluster, y7_test, True, 'markov_improved.txt')
                    
if __name__ == '__main__':
   main()