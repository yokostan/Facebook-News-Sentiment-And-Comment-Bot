# this file is for models to perform sentiment_analysis
import heapq
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score

class StochasticGradientDescent:
    def __init__(self):
        self.count_vect = CountVectorizer()

    def train(self, X_train, y_train, X_val, y_val, losses=None, learning_rates=None, tols=None, max_iter=None):
        def get_param(param, default):
            return param if param is not None else default   
        X_train = self.count_vect.fit_transform(['' if x is np.nan else x for x in X_train])
        if len(X_val) > 0: # we have val data to tune the params
            max_accuracy = 0
            current_best_model = None
            index = 0
            losses = get_param(losses,["hinge"])
            learning_rates = get_param(learning_rates,["optimal"])
            tols = get_param(tols,[1e-3])
            for loss in losses:
                for learning_rate in learning_rates:
                    for tol in tols:
                        print("Number "+str(index)+" training started")
                        model = SGDClassifier(loss=loss, learning_rate=learning_rate, penalty="l2", max_iter=get_param(max_iter,500), tol=tol)
                        model.fit(X_train, y_train)
                        y_val_pred = model.predict(self.count_vect.transform(['' if x is np.nan else x for x in X_val]))
                        accuracy = accuracy_score(y_val, y_val_pred)
                        if accuracy > max_accuracy:
                            current_best_model = model
                            max_accuracy = accuracy    
                        index += 1             
        else:
            print("training started")
            current_best_model = SGDClassifier(loss="hinge", penalty="l2", max_iter=500, tol=1e-3)
            current_best_model.fit(X_train, y_train)
            y_val_pred = current_best_model.predict(self.count_vect.transform(['' if x is np.nan else x for x in X_val]))
            max_accuracy = accuracy_score(y_val, y_val_pred)
        print('The parameters of the best model are: ')
        print(current_best_model)
        print('The validation accuracy is '+str(max_accuracy))
        return current_best_model

    def predict(self, model, X_test, y_test, filename):
        y_pred = model.predict(self.count_vect.transform(['' if x is np.nan else x for x in X_test]))
        df = pd.DataFrame ({'message': X_test,'true_label': y_test,'pred_label': y_pred})
        df.to_csv(filename+'.csv', index=False)  
        print('The prediction accuracy is '+str(accuracy_score(y_test, y_pred)))
        print('---------------------------------------------------------------------------')

class KMeansModel():
    def __init__(self):
        self.count_vect = CountVectorizer()

    def categorize_reaction(self, reactions):
        max_reaction_list = []
        for i in range(len(reactions[0])):
            max_reaction_idx = None
            max_reaction_count = 0
            for j in range(len(reactions)): # for each of the 6 reactions
                if reactions[j][i] > max_reaction_count:
                    max_reaction_count = reactions[j][i]
                    max_reaction_idx = j 
            max_reaction_list.append(max_reaction_idx) 
        return max_reaction_list

    def clusters_mapping(self, y_category, y_pred):
            cluster_cluster_map = {}
            for i in range(len(y_category)):
                temp_list = [[0,0]]*len(y_pred)
                if y_pred[i] not in cluster_cluster_map:
                    temp_list = [[y_category[i], 1]]
                else:
                    value_list = cluster_cluster_map.get(y_pred[i])
                    for j in range(len(value_list)):
                        if value_list[j][0] != y_category[i]:
                            temp_list[j] = value_list[j]
                        else:
                            temp_list[j] = [value_list[j][0], value_list[j][1]+1]
                cluster_cluster_map[y_pred[i]] = temp_list
            
            cluster_index_map = {}
            for key in cluster_cluster_map:
                value_list = cluster_cluster_map.get(key)
                max_count = 0
                max_cluster_index = None
                for value in value_list:
                    if value[1] > max_count:
                        max_count = value[1]
                        max_cluster_index = value[0]
                cluster_index_map[key] = max_cluster_index  
                
            return [cluster_index_map.get(y_pred[i]) for i in range(len(y_pred))]

    def get_accuracy(self, y_true, y_pred):
        count = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                count += 1
        return count/len(y_true)

    def train(self, X_train, X_val, y_val_list, n_clusters_list=None, tols=None, max_iter=None):
        def get_param(param, default):
            return param if param is not None else default  

        X_train = self.count_vect.fit_transform(['' if x is np.nan else x for x in X_train]) 
        y_val_category = self.categorize_reaction(y_val_list)
        if len(X_val) > 0: # we have val data to tune the params
            max_accuracy = -1e5
            current_best_model = None
            index = 0
            n_clusters_list = get_param(n_clusters_list, [6])
            tols = get_param(tols,[1e-3])
            for n_clusters in n_clusters_list:
                for tol in tols:
                    print("Number "+str(index)+" training started")
                    model = KMeans(n_clusters=n_clusters, max_iter=get_param(max_iter,300), tol=tol)
                    model.fit(X_train)
                    y_val_pred = model.predict(self.count_vect.transform(['' if x is np.nan else x for x in X_val]))
                    y_val_pred = self.clusters_mapping(y_val_category, y_val_pred)
                    accuracy = self.get_accuracy(y_val_category, y_val_pred)
                    if accuracy > max_accuracy:
                        current_best_model = model
                        max_accuracy = accuracy
                    index += 1             
        else:
            print("training started")
            current_best_model = KMeans(n_clusters=6, max_iter=300, tol=1e-4)
            current_best_model.fit(X_train)
        print('The parameters of the best model are: ')
        print(current_best_model)
        if len(X_val) > 0:
            print('The validation accuracy is '+str(max_accuracy))
        return current_best_model

    def predict(self, model, X_test, y_test, y_test_list, filename):
        y_test_category = self.categorize_reaction(y_test_list)
        y_pred = model.predict(self.count_vect.transform(['' if x is np.nan else x for x in X_test]))
        y_pred = self.clusters_mapping(y_test_category, y_pred)
        df = pd.DataFrame ({'message': X_test,'true_react_angry': y_test[0], 'true_react_haha': y_test[1], 'true_react_like': y_test[2], 'true_react_love': y_test[3], 'true_react_sad': y_test[4], 'true_react_wow': y_test[5], \
            'true_cluster_id': y_test_category, 'pred_cluster_id': y_pred})
        df.to_csv(filename+'.csv', index=False)  
        accuracy = self.get_accuracy(y_test_category, y_pred)
        print('The prediction accuracy is '+str(accuracy))
        print('---------------------------------------------------------------------------')