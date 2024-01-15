#!/usr/bin/env python3
import random

def generate_scores(number, elements, seed = 42):
    def get_random():
        return random.uniform(0.0, 1.0)
    random.seed(seed)
    return [tuple(get_random() for y in range(elements)) for x in range(number)]

def adjudication(value):
    if value < 0.2: return 0
    elif value < 0.4: return 1
    elif value < 0.6: return 2
    elif value < 0.8: return 3
    else: return 4

def score_larger(scores):
    return sum(adjudication(z) for z in scores)

def evaluate_larger_scores(probs):
    return [(score_larger(x), x) for x in probs]

larger_probabilities = generate_scores(100, 8)
larger_scores = evaluate_larger_scores(larger_probabilities)
print(larger_scores)