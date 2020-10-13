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

def train_test_val_split(X, y, train_ratio, test_ratio, val_ratio=0):
    # train test ratio should sum up to 1 if val ratio is not provided
    # otherwise, the three ratios should sum up to 1
    if train_ratio + test_ratio + val_ratio - 1 > 1e-5:
        print("please provide eligible train, test, val ratio")
        return None, None, None, None, None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=1)
        if val_ratio != 0:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio/(train_ratio+val_ratio), random_state=1) # test_size = second split ratio between train and val
        return X_train, y_train, X_test, y_test, X_val, y_val

def main():
    post_df, comment_df, train_ratio, test_ratio, val_ratio = read_csv_to_df()
    # sentiment analysis
    post_X_train, post_y_train, post_X_test, post_y_test, post_X_val, post_y_val = train_test_val_split(post_df['message'].tolist(), post_df['label'].tolist(), train_ratio, test_ratio, val_ratio)
    if all(v is not None for v in [post_X_train, post_y_train, post_X_test, post_y_test]):
        # method 1(baseline): sgd model
        sgd = StochasticGradientDescent()
        model = sgd.train(post_X_train, post_y_train, post_X_val, post_y_val, losses=['hinge', 'log', 'squared_loss'], learning_rates=['optimal'], tols=[1e-5, 1e-4, 1e-3], max_iter=500)
        sgd.predict(model, post_X_test, post_y_test)


if __name__ == '__main__':
   main()