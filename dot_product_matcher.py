__author__ = 'kafuinutakor'

from textmining import simple_tokenize_remove_stopwords as tokenize, stem, TermDocumentMatrix
import numpy as np
import operator


class DotProductMatcher:
    def __init__(self):
        self.terms = {}  # dict of search terms; read in from file

        with open('terms.txt', 'r') as infile:
            for num, line in enumerate(infile):
                self.terms[line.splitlines()[0]] = []  # bucket for meta data;
        self.terms_flat = [stem(term) for phrase in self.terms for term in phrase.split()]
        self.doc_ids = []  # maps index_pos to _id
        self.tdm = TermDocumentMatrix()
        self.term_index_lookup = {}  # lookup table of index positions for terms on TDM
        self.matrix = np.array([])  # nd array structure for TDM
        self.out_dict = {}  # output dict; comp site for phrase matching
        self.term_index_dict = {}  # pseudo hash trick dict

    def index(self, (doc_id, text)):
        """
        processes text into TDM and tracks doc id
        """
        self.doc_ids.append(doc_id)  # map index position to doc id without hash
        tokens = set([stem(i) for i in tokenize(text)]).intersection(self.terms_flat)  # tokenize and stem
        self.tdm.add_doc(' '.join(list(tokens)))

    def convert_to_array(self):
        """
        converts textmining.tdm object to nd array
        """
        self.matrix = np.array([i for i in self.tdm.rows(cutoff=1)])

    def create_columns_dict(self):
        """
        maps terms to index positions; pseudo hash trick
        """
        for column in xrange(0, len(self.matrix[0, :])):  # term row from TDM
            self.term_index_dict[self.matrix[:, column][0]] = column  # term: <index_position>

    def compute_phrase_matches(self):
        """
        computes phrase matches as dot products of term vectors
        """
        for search_phrase in self.terms.keys():
            terms = [stem(term) for term in search_phrase.split()]  # add stemming
            if len(terms) == 1:  #retrieve column vector for unigrams
                try:
                    self.out_dict[search_phrase] = self.matrix[:, dict[search_phrase]][1:]  # grab column for term and skip first row
                except (KeyError, TypeError):
                    pass
            else:  # get vectors to multiply; compute dot product of vectors
                try:
                    dot_product = reduce(operator.__mul__, [self.matrix[:, self.term_index_dict[a]][1:].astype(int) for a in terms], 1)
                    self.out_dict[search_phrase] = dot_product
                except (KeyError, TypeError):
                    pass

            tags = {}
            for search_phrase in self.out_dict:
                # maintain original doc index pos with enumeration
                reduced = [a for a in enumerate(self.out_dict[search_phrase]) if a[1] != 0 and a[1] != '0']
                for pair in reduced:
                    tags.setdefault(pair[0], [])
                    tags[pair[0]].append(search_phrase)  # doc_index_pos: tagged_phrase
            doc_lookup = {}
            for _id in enumerate(self.doc_ids):
                doc_lookup[_id[0]] = _id[1]
            #out
            out = {}
            for search_phrase in tags:
                out[doc_lookup[search_phrase]] = tags[search_phrase]
            matches = [[i[0], ','.join(i[1])] for i in out.iteritems()]  # doc id and list of phrase matches
        return matches
