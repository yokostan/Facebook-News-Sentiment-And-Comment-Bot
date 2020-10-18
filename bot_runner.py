# this is the bot_runner.py file for running the bot on root level
# the bot_runner is composed of two part
#   part 1: sentiment analysis
#   part 2: comment generation
import pandas as pd
from sklearn.model_selection import train_test_split
from model.sentiment_analysis import StochasticGradientDescent

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

def train_test_val_split(df, type, train_ratio, test_ratio, val_ratio=0):
    # train test ratio should sum up to 1 if val ratio is not provided
    # otherwise, the three ratios should sum up to 1
    if train_ratio + test_ratio + val_ratio - 1 > 1e-5:
        print("please provide eligible train, test, val ratio")
        return [[]*24]
    else:
        if type == 'post':
            X = df['message'].to_list()
            y0 = df['label'].to_list()
            y1 = df['react_angry'].to_list()
            y2 = df['react_haha'].to_list()
            y3 = df['react_like'].to_list()
            y4 = df['react_love'].to_list()
            y5 = df['react_sad'].to_list()
            y6 = df['react_wow'].to_list()
            X_train, X_test, y0_train, y0_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test, \
            y4_train, y4_test, y5_train, y5_test, y6_train, y6_test = train_test_split(X, y0, y1, y2, y3, y4, y5, y6, test_size=test_ratio, random_state=1)
            X_val = y0_val = y1_val = y2_val = y3_val = y4_val = y5_val = y6_val = None
            if val_ratio != 0:
                X_train, X_val, y0_train, y0_val, y1_train, y1_val, y2_train, y2_val, y3_train, y3_val, \
                y4_train, y4_val, y5_train, y5_val, y6_train, y6_val = train_test_split(X_train, y0_train, y1_train, y2_train, y3_train, \
                    y4_train, y5_train, y6_train, test_size=val_ratio/(train_ratio+val_ratio), random_state=1) # test_size = second split ratio between train and val
            return [X_train, X_test, y0_train, y0_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test, y5_train, y5_test, \
            y6_train, y6_test, X_val, y0_val, y1_val, y2_val, y3_val, y4_val, y5_val, y6_val]

def main():
    post_df, comment_df, train_ratio, test_ratio, val_ratio = read_csv_to_df()
    # problem 1: sentiment analysis
    # reminder: the data split is different each time we run the program
    #           so the models are only comparable from the same run
    post_data_list = train_test_val_split(post_df, 'post', train_ratio, test_ratio, val_ratio)
    if post_data_list is not None:
        # method 1: sgd model
        sgd = StochasticGradientDescent()
        # baseline: using default values for parameters
        model = sgd.train(post_data_list[0], post_data_list[2], post_data_list[16], post_data_list[17])
        sgd.predict(model, post_data_list[1], post_data_list[3], 'sgd_baseline')
        # sgd model with tuned paramters
        model = sgd.train(post_data_list[0], post_data_list[2], post_data_list[16], post_data_list[17], losses=['hinge', 'log', 'squared_loss'], learning_rates=['optimal'], tols=[1e-5, 1e-4, 1e-3], max_iter=500)
        sgd.predict(model, post_data_list[1], post_data_list[3], 'sgd_tuned')
        
if __name__ == '__main__':
   main()