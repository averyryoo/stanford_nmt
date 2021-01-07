#Code Source: http://www.nltk.org/_modules/nltk/align/bleu.html

from __future__ import division

import math

from nltk import word_tokenize
from nltk.compat import Counter
from nltk.util import ngrams

def compute_BLEU(candidate, references, weights):
    candidate = [c.lower() for c in candidate]
    references = [[r.lower() for r in reference] for reference in references]

    p_ns = (BLEU.modified_precision(candidate, references, i) for i, _ in enumerate(weights, start=1))
    s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)

    bp = BLEU.brevity_penalty(candidate, references)
    return bp * math.exp(s)

def modified_precision(candidate, references, n):
    counts = Counter(ngrams(candidate, n))

    if not counts:
        return 0

    max_counts = {}
    for reference in references:
        reference_counts = Counter(ngrams(reference, n))
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

    clipped_counts = dict((ngram, min(count, max_counts[ngram])) for ngram, count in counts.items())

    return sum(clipped_counts.values()) / sum(counts.values())

def brevity_penalty(candidate, references):
    c = len(candidate)
    r = min(abs(len(r) - c) for r in references)

    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)