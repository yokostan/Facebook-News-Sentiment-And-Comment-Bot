# this file is for models to perform comment_generation
import markovify
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import time

class Markov():
    def __init__(self, post_train, post_test, comment_list_train, comment_list_test):
        vocabulary = post_train + post_test
        for comment in comment_list_train:
            if isinstance(comment, str):
                vocabulary.append(comment)
        for comment in comment_list_test:
            if isinstance(comment, str):
                vocabulary.append(comment)
        self.tfidf = TfidfVectorizer().fit(['' if x is np.nan else x for x in vocabulary])   

    def add_comment_to_text(self, comment_list, text_dict, cluster_id):
        markov_text = text_dict.get(cluster_id, '')
        for comment in comment_list:
            if isinstance(comment, str):
                markov_text += ' ' + comment
        return markov_text

    def generate_markov_dict(self, posts, cluster_ids, comment_list):
        print('generating markov dict ...')
        markov_text_dict = {}
        for i in range(len(posts)):
            cluster_id = cluster_ids[i]
            markov_text_dict[cluster_id] = self.add_comment_to_text(comment_list[i], markov_text_dict, cluster_id)
            if i % 100 == 0:
                print('post ' + str(i) + '\'s word dict merged')
        return markov_text_dict

    def get_text_similarity(self, gen_comment, comment_list):
        comment_text = ''
        for comment in comment_list:
            if isinstance(comment, str):
                comment_text += ' ' + comment
        tfidf = self.tfidf.transform([gen_comment]+[comment_text])
        return (tfidf * tfidf.T).toarray()[0][1]

    def generate_comment(self, markov_dict, posts, cluster_ids, comment_list, pos_tag_flag, filename):
        # to diversify the generated comments between same cluster
        # we will add some amount of post vocabulary to the markov dict
        print('generating comments ...')
        gen_comments = []
        text_similarities = []
        start_time = time.time()
        test_range = 100
        for i in range(test_range): #range(len(posts)): # it takes 3s for one post and ~25 hr for all 3000 test post
            combined_text = markov_dict.get(cluster_ids[i], '') + posts[i]
            gen_comment = ''
            try:
                if pos_tag_flag is False:
                    text_model = markovify.Text(combined_text)
                else:
                    text_model = markovify.combine([markovify.Text(markov_dict.get(cluster_ids[i], '')), markovify.Text(posts[i])], [2, 1])
                gen_comment = text_model.make_sentence()
                gen_comment = gen_comment if gen_comment is not None else ''
                text_similarity = self.get_text_similarity(gen_comment, comment_list[i])
                text_similarities.append(text_similarity)
            except:
                gen_comment = ''
            gen_comments.append(gen_comment)
            print('-------------------------------------------')
            print('post is:')
            print(posts[i])
            print('.....below is generated comment.....')
            print(gen_comment)
            print('-------------------------------------------')
            if i % 10 == 0:
                if i == 0:
                    print("--- %s seconds for 1 post ---" % (time.time() - start_time))
                else:
                    print("--- %s seconds in total ---" % (time.time() - start_time))
                print('post ' + str(i) + '\'s comment generated')
        
        with open(filename, 'w') as f:
            for item in gen_comments:
                f.write("%s\n" % item)
        
        text_similarity = sum(text_similarities)/len(text_similarities) if len(text_similarities) != 0 else 0
        print('The text similarity is '+str(text_similarity))
        print('The text generation rate is '+str(len(text_similarities)/test_range)) #/len(posts))