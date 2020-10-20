# this is the bot_runner.py file for running the bot on root level
# the bot_runner is composed of two part
#   part 1: sentiment analysis
#   part 2: comment generation
import csv
import difflib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

def train_test_val_split(df, type, train_ratio, test_ratio, val_ratio=0):
    # train test ratio should sum up to 1 if val ratio is not provided
    # otherwise, the three ratios should sum up to 1
    if train_ratio + test_ratio + val_ratio - 1 > 1e-5:
        print("please provide eligible train, test, val ratio")
        return []*24
    else:
        if type == 'post':
            X = df['message'].to_list()
            # y0 = df['label'].to_list()
            y1 = df['react_angry'].to_list()
            y2 = df['react_haha'].to_list()
            y3 = df['react_like'].to_list()
            y4 = df['react_love'].to_list()
            y5 = df['react_sad'].to_list()
            y6 = df['react_wow'].to_list()
            y0 = get_sentiment([y1, y2, y3, y4, y5, y6])
            X_train, X_test, y0_train, y0_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test, \
            y4_train, y4_test, y5_train, y5_test, y6_train, y6_test = train_test_split(X, y0, y1, y2, y3, y4, y5, y6, test_size=test_ratio, random_state=1)
            X_val = y0_val = y1_val = y2_val = y3_val = y4_val = y5_val = y6_val = None
            if val_ratio != 0:
                X_train, X_val, y0_train, y0_val, y1_train, y1_val, y2_train, y2_val, y3_train, y3_val, \
                y4_train, y4_val, y5_train, y5_val, y6_train, y6_val = train_test_split(X_train, y0_train, y1_train, y2_train, y3_train, \
                    y4_train, y5_train, y6_train, test_size=val_ratio/(train_ratio+val_ratio), random_state=1) # test_size = second split ratio between train and val
            return [X_train, X_val, y0_train, y0_val, y1_train, y2_train, y3_train, y4_train, y5_train, y6_train, \
                y1_val, y2_val, y3_val, y4_val, y5_val, y6_val, X_test, y0_test, y1_test, y2_test, y3_test, y4_test, y5_test, y6_test]
        else:
            X = [row[0] for row in df]
            y = [row[1] for row in df]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=1)
            X_val = y_val = None
            if val_ratio != 0:
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio/(train_ratio+val_ratio), random_state=1) # test_size = second split ratio between train and val
            return [X_train, X_val, X_test, y_train, y_val, y_test]

def generate_post_comment_nested_dict(post, comment, n):
    def ngrams(comment, n):
        output = []
        if isinstance(comment, str):
            comment = comment.split(' ')
            for i in range(len(comment)-n+1):
                output.append(comment[i:i+n])
        return output

    # merge post and comment df and only take the messsage and comment columns 
    comment = comment.rename(columns={"message": "comment", "post_id": "post_id_1"})
    new_df = pd.merge(post, comment, left_on='post_id', right_on='post_id_1', how='left')[['message', 'comment']]
    
    # perform n gram to generate word pairs for words in comments
    post_comment_dict = {}
    for index, row in new_df.iterrows():
        post = row['message']
        ngram_list = ngrams(row['comment'], 2)
        # construct a nested dict where the key is post 
        # and the value is a word dict for n-gram words from its comments
        word_dict = post_comment_dict.get(post, {})
        for word_1, word_2 in ngram_list:
            if word_1 in word_dict.keys():
                word_dict[word_1].append(word_2)
            else:
                word_dict[word_1] = [word_2]
        post_comment_dict[post] = word_dict
    with open('post_comment_dict.csv', 'w') as f:
        for key in  post_comment_dict.keys():
            f.write("%s,%s\n"%(key, post_comment_dict[key]))
    return post_comment_dict

def transform_dict_to_list(data_dict):
    data_list = []
    for key in data_dict:
        word_pairs = []
        word_dict = data_dict.get(key)
        for word_1 in word_dict:
            for word_2 in word_dict.get(word_1):
                word_pair = (' ').join([word_1, word_2]) #word_pair = "word1 word2"
                word_pairs.append(word_pair)
        data_list.append([key, word_pairs])
    return data_list

def main():
    post_df, comment_df, train_ratio, test_ratio, val_ratio = read_csv_to_df()
    # switch for two problems
    sentiment_analysis = False
    comment_generation = True
    # problem 1: sentiment analysis
    # reminder: the data split is different each time we run the program
    #           so the models are only comparable from the same run
    post_df = remove_null_rows(post_df)
    post_data_list = train_test_val_split(post_df, 'post', train_ratio, test_ratio, val_ratio)
    if post_data_list is not None and sentiment_analysis is True:
        X_train = post_data_list[0]
        X_val = post_data_list[1]
        y0_train = post_data_list[2]
        y0_val = post_data_list[3]
        X_test = post_data_list[16]
        y0_test = post_data_list[17]

        # baseline: gaussian naive bayes
        gnb = GNBModel()
        model = gnb.train(X_train, y0_train)
        gnb.predict(model, X_test, y0_test, 'gnb')

        # method 1: kmeans model
        kmeans = KMeansModel()
        model = kmeans.train(X_train, X_val, y0_val, post_data_list[10:16], n_clusters_list=[6, 12, 18, 24], tols=[1e-6, 1e-5, 1e-3], max_iters=[300, 500])
        kmeans.predict(model, X_test, y0_test, post_data_list[18:], 'kmeans')
        
        # method 2: sgd model
        sgd = StochasticGradientDescent()
        model = sgd.train(X_train, y0_train, X_val, y0_val, losses=['hinge', 'log', 'squared_loss'], learning_rates=['optimal'], tols=[1e-5, 1e-4, 1e-3], max_iter=500)
        sgd.predict(model, X_test, y0_test, 'sgd')

        # method 3: svm model 
        svm = SVMModel()
        model = svm.train(X_train, y0_train, X_val, y0_val, kernels=['linear', 'poly', 'rbf', 'sigmoid'], gammas=['auto', 'scale'], tols=[1e-5, 1e-4, 1e-3], max_iters=[300, 500])
        svm.predict(model, X_test, y0_test, post_data_list[18:], 'svm')

    # problem 2: comment generation
    # comment_df = remove_null_rows(comment_df) # removed because this takes time and there is no null row in comment
    post_comment_dict = generate_post_comment_nested_dict(post_df, comment_df, 2)
    post_comment_dict_list = transform_dict_to_list(post_comment_dict)
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(post_comment_dict_list, 'comment', train_ratio, test_ratio, val_ratio)

if __name__ == '__main__':
   main()