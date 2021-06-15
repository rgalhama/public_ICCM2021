"""
Author : Raquel G. Alhama
Desc:
"""
#import toolkits.hyperwords_f5a01ea3e44c.hyperwords.representations.representation_factory as reprf
from toolkits.hyperwords_f5a01ea3e44c.hyperwords.representations.embedding import SVDEmbedding, EnsembleEmbedding, Embedding

class ReprWrapper():


    def __init__(self, model, path):
        self.representation = self.load_representation(model, path)

    def load_representation(self, model, path):

        if model.lower() == "sgns":
            self.representation = Embedding(path + 'sgns.words', True)
            print("Representation loaded.")
        else:
            self.representation = SVDEmbedding(path + 'svd', True)
            print("Representation loaded.")
        #self.representation = reprf.create_representation(args)

        return self.representation

    def get_similarity(self, w1, w2):
        similarity = self.representation.similarity(w1, w2)
        return similarity
