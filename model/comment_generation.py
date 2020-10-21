# this file is for models to perform comment_generation
import markovify

class Markov():
    def add_comment_to_text(self, comment_list, text_dict, post):
        markov_text = text_dict.get(post, '')
        for comment in comment_list:
            if isinstance(comment, str):
                markov_text += comment
        return markov_text
    def generate_markov_text(self, post, cluster_id, comment_list):
        markov_text_dict = {}
        for i in range(20): #(len(cluster_id)):
            cluster_index = cluster_id[i]
            markov_text_dict[cluster_index] = self.add_comment_to_text(comment_list[i], markov_text_dict, post[i])
            if i % 100 == 0:
                print('post ' + str(i) + '\'s word dict merged')

        for value in markov_text_dict.values():
            text_model = markovify.Text(value, state_size = 2, well_formed=False)
            print(text_model.make_sentence())
        return markov_text_dict
