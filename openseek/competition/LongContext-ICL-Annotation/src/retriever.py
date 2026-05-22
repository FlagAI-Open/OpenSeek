import re
from rank_bm25 import BM25Okapi

def _english_tokenize(text):
    """Simple English tokenizer: lowercase + split on non-alphanumeric characters."""
    return re.findall(r'[a-z0-9]+', text.lower())

class BM25Retriever:
    def __init__(self, examples):
        self.examples = examples
        corpus = [ex['input'] for ex in examples]
        self.tokenized_corpus = [_english_tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def retrieve_top_k(self, query, top_k=5):
        tokenized_query = _english_tokenize(query)
        return self.bm25.get_top_n(tokenized_query, self.examples, n=top_k)
