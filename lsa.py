from glob import glob
from collections import Counter
import os.path as osp
import numpy as np
import math, re, logging, itertools, sys

class Bow(object):
    def __init__(self, path=None):
        self.path = path
        self.bow = None
        if self.path:
            self.__parse()
        self.tfidf = None
        self.empty = False if len(self.bow.keys()) else True

    def __parse(self):
        with open(self.path, 'r') as doc_file:
            self.bow = dict(Counter(                   \
                    itertools.chain.from_iterable(     \
                    re.findall(r"[\w]+", line.lower()) \
                    for line in doc_file)))

class Lsa(object):

    def __init__(self, path=None, nb_topics=10, *file_ext):
        self.path = path
        self.nb_topics = nb_topics
        self.file_ext = file_ext
        self.voc = set()
        self.dnames = None
        self.docs = None
        self.tfidf = None
        self.Uk = None
        self.Vk = None
        self.Sk = None
        self.build()

    def __str__(self):
        return "Lsa (path = %s, vocabulary size = %i)" % (self.path, len(self.voc))

    def __fetch_doc_paths(self):
        if len(self.file_ext):
            self.dnames = list()
            for ext in self.file_ext:
                self.dnames += glob(osp.join(self.path, '*.%s' % ext))
        else:
            self.dnames = glob(osp.join(self.path, '*.*'))
        logging.info("%i documents under %s", len(self.dnames), self.path)

    def build(self):
        self.gen_docs()
        self.gen_voc()
        self.gen_tfidf()
        self.svd_k(self.nb_topics)

    def gen_docs(self):
        self.__fetch_doc_paths()
        self.docs = [Bow(dname) for dname in self.dnames]

    def gen_voc(self):
        for doc in self.docs:
            self.voc = self.voc.union(doc.bow.keys())
        self.rev_voc = {word_idx: word for word_idx, word in enumerate(self.voc)}

    def gen_tfidf(self):
        self.tfidf = np.zeros(shape=(len(self.voc), len(self.docs)))
        highest_count = float(max([max(doc.bow.values()) for doc in self.docs]))
        for doc_idx, doc in enumerate(self.docs):
            for word_idx, word in enumerate(self.voc):
                self.tfidf[word_idx , doc_idx] = doc.bow.get(word, 0)/highest_count
        for word_idx, word in enumerate(self.voc):
            norm = float(sum(1 for doc in self.docs if doc.bow.get(word, 0)))
            self.tfidf[word_idx] *= math.log(len(self.docs)/norm)

    def svd_k(self, k):
        U, S, V = np.linalg.svd(self.tfidf)
        if k > len(S):
            raise ValueError("given k (%i) is higher than the dimension of S (%i)" % (k, len(S)))
        Sk = np.identity(k)
        for i in xrange(0, k):
            Sk[i] *= S[i]
        self.Uk = U[:,0:k]
        self.Vk = V[0:k]
        self.Sk = Sk

    def concepts(self, nb_words):
        concepts = list()
        matrix = np.dot(np.linalg.inv(self.Sk), self.Uk.transpose())
        for line in matrix:
            threshold = sorted(line)[nb_words]
            concept = [(self.rev_voc[idx], item) for idx, item \
                    in enumerate(line) if item < threshold]
            concept.sort(key = lambda x : x[1])
            concepts.append(concept)
        return concepts

if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    logging.basicConfig(format='%(message)s')
    lsa = Lsa(sys.argv[1], int(sys.argv[2]), 'txt')
    c = lsa.concepts(10)
    for concept in c:
        print [con[0] for con in concept]
