#!/usr/bin/env python3
"""
Building Machine Learning Powered Applications Chapter 3: Build Your First End-to-End Pipeline
"""
import argparse
import nltk
nltk.download('punkt_tab')
import pyphen

def parse_arguments():
    '''
    returns:
        The text to be edited
    '''
    parser = argparse.ArgumentParser(description = "Receives the text to be edited")
    parser.add_argument('text', metavar = 'input text', type = str)
    args = parser.parse_args()
    return args.text

def clean_input(text):
    '''
    params:
        text - User input text
    returns:
        Sanitized text, without non-ASCII characters
    '''
    # Keep only ASCII characters for simplicity
    return str(text.encode().decode('ascii', errors = 'ignore'))

def preprocess_input(text):
    '''
    params:
        text - Sanitized text
    returns:
        Text ready to be fed for analysis with sentences and words tokenized
    '''
    sentences = nltk.sent_tokenize(text)
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return tokens

def count_word_usage(tokens, word_list):
    '''
    Counts the number of occurrences for a given list of words
    params:
        tokens - A list of tokens for a given sentence
        word_list - A list of words to search for
    returns:
        The number of times the words appear in the list
    '''
    return len([word for word in tokens if word.lower() in word_list])

def calculate_average_word_length(tokens):
    '''
    Calculates the word length of a sentence
    params:
        tokens - A list of words
    returns:
        The average length of words in this list of tokens
    '''
    word_lengths = [len(word) for word in tokens]
    return sum(word_lengths)/len(word_lengths)

def calculate_total_average_word_length(sentence_list):
    '''
    Calculates the average word length over multiple sentences
    params:
        sentence_list - A list of sentences, where each sentence is a list of words
    returns:
        The average length of words in this list of sentences
    '''
    sentence_lengths = [calculate_average_word_length(tokens) for tokens in sentence_list]
    return sum(sentence_lengths)/len(sentence_lengths)

def calculate_total_unique_words_fraction(sentence_list):
    '''
    Calculates the fraction of unique words
    params:
        sentence_list - A list of sentences, where each sentence is a list of words
    return:
        The fraction of unique words in this list of sentences
    '''
    all_words = [word for word_list in sentence_list for word in word_list]
    unique_words = set(all_words)
    return len(unique_words)/len(all_words)

def count_word_syllables(word):
    '''
    Counts the number of syllables in each word
    params:
        word - A one word string
    returns:
        The number of syllables according to the pyphen package
    '''
    dictionary = pyphen.Pyphen(lang = 'en_US')
    hyphenated = dictionary.inserted(word)
    return len(hyphenated.split('-'))

def count_sentence_syllables(tokens):
    '''
    Counts the number of syllables in each sentence
    params:
        tokens - A list of words and potentially punctuations
    returns:
        The number of syllables in the sentence
    '''
    # The tokenizer leaves punctuations as separate words, so they need to filtered out
    punctuations = '.,!?/'
    return sum([count_word_syllables(word) for word in tokens if word not in punctuations])

def count_total_syllables(sentence_list):
    '''
    Counts the number of syllables in a list of sentences
    params:
        sentence_list - A list of sentences, where each sentence is a list of words
    returns:
        The number of syllables in this list of sentences
    '''
    return sum([count_sentence_syllables(sentence) for sentence in sentence_list])

def count_words_per_sentence(tokens):
    '''
    Counts the number of words in a sentence
    params:
        tokens - A list of words and potentially punctuations
    returns:
        The number of words in the sentence
    '''
    punctuations = ".,!?/"
    return len([word for word in tokens if word not in punctuations])

def count_total_words(sentence_list):
    '''
    Counts the number of words in a list of sentences
    params:
        sentence_list - A list of sentences, where each sentence is a list of words
    returns:
        The number of words in this list of sentences
    '''
    return sum([count_words_per_sentence(sentence) for sentence in sentence_list])

def calculate_flesch_reading_ease_score(total_words, total_sentences, total_syllables):
    '''
    Calculates the readability score from the summary statistics
    params:
        total_words - The number of words in the input text
        total_sentences - The number of sentences in the input text
        total_syllables - The number of syllables in the input text
    returns:
        The Flesch reading-ease score (FRES), where the lower the score implies that the text is
        deemed to be more difficult to read
    '''
    return 206.835 - (1.015 * (total_words/total_sentences)) - (84.6 * (total_syllables/total_words))

def get_flesch_readibility_level(flesch_score):
    '''
    Source: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease
    params:
        flesch_score
    returns:
        A readability level or level of difficulty for the given flesch_score
    '''
    if flesch_score <= 30.0:
        return "Very difficult to read"
    elif flesch_score <= 50.0:
        return "Difficult to read"
    elif flesch_score <= 60.0:
        return "Fairly difficult to read"
    elif flesch_score <= 70.0:
        return "Plain English"
    elif flesch_score <= 80.0:
        return "Fairly easy to read"
    elif flesch_score <= 90.0:
        return "Easy to read"
    else:
        return "Very easy to read"

def get_suggestions(sentence_list):
    '''
    Returns a string containing our suggestions
    params:
        sentence_list - A list of sentences, where each sentence is a list of words
    returns:
        Suggestions to improve the input text
    '''
    told_said_usage = sum((count_word_usage(tokens, ['told', 'said']) for tokens in sentence_list))
    but_and_usage = sum((count_word_usage(tokens, ['but', 'and']) for tokens in sentence_list))
    wh_adverbs_usage = sum((count_word_usage(tokens, ['when', 'where', 'why', 'whence', 'whereby',
                                                      'wherein', 'whereupon'])
                            for tokens in sentence_list))

    result_str = ""
    adverb_usage = "Adverb usage: {} told/said, {} but/and, {} wh adverbs".format(
        told_said_usage, but_and_usage, wh_adverbs_usage)
    result_str += adverb_usage
    result_str += "<br/>"

    average_word_length = calculate_total_average_word_length(sentence_list)
    unique_words_fraction = calculate_total_unique_words_fraction(sentence_list)
    word_stats = "Average word length: {:.2f}, Fraction of unique words: {:.2f}".format(
        average_word_length, unique_words_fraction)
    # Using a HTML break for a webapp display
    result_str += word_stats
    result_str += "<br/>"

    number_of_syllables = count_total_syllables(sentence_list)
    number_of_words = count_total_words(sentence_list)
    number_of_sentences = len(sentence_list)
    syllable_counts = "{:d} syllables, {:d} words, {:d} sentences".format(
        number_of_syllables, number_of_words, number_of_sentences)
    result_str += syllable_counts
    result_str += "<br/>"

    flesch_score = calculate_flesch_reading_ease_score(number_of_words, number_of_sentences,
                                                       number_of_syllables)
    flesch = '{} syllables, Flesch score: {:.2f} ({})'.format(
        number_of_syllables, flesch_score, get_flesch_readibility_level(flesch_score))
    result_str += flesch

    return result_str

def get_recommendations_from_input(text):
    '''
    Cleans and preprocesses the input string, and finally returns a heuristic suggestion for the string
    params:
        text - Input string
    returns:
        Suggestions for cleaned and preprocessed input string
    '''
    cleaned_text = clean_input(text)
    tokenized_sentences = preprocess_input(cleaned_text)
    suggestions = get_suggestions(tokenized_sentences)
    return suggestions

if __name__ == "__main__":
    input_text = parse_arguments()
    print(get_recommendations_from_input(input_text))