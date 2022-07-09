import re
import nltk
import string
from nltk.stem import WordNetLemmatizer
import pickle


def clean_review(review):

    stops = []

    step1 = "".join([i for i in review if i not in string.punctuation])

    step2 = "".join([i.lower() for i in step1])

    step3 = re.sub('[^A-Za-z0-9.]+', ' ', step2)

    step4 = re.sub(r'http\S+', '', step3)

    words = nltk.tokenize.word_tokenize(step4)

    stopwords = nltk.corpus.stopwords.words('english')
    [stops.append(i) for i in words if i in stopwords]
    step5 = [i for i in words if i not in stopwords]

    lemmatizer = WordNetLemmatizer()
    review_cleaned = [lemmatizer.lemmatize(word) for word in step5]

    review_cleaned = " ".join(review_cleaned)

    return review_cleaned, stops


def naive_bayes_predict(review, logprior, loglikelihood):

    tokens = {}

    word_l, stops = clean_review(review)
    word_l = word_l.split()

    total_prob = 0
    total_prob = total_prob + logprior

    for word in word_l:
        if word in loglikelihood:
            tokens[word] = loglikelihood[word]
            total_prob = total_prob + loglikelihood[word]

    for word in stops:
        tokens[word] = 0

    if total_prob > 0:
        total_prob = 1
    else:
        total_prob = 0

    return total_prob, tokens


if __name__ == "__main__":

    while 1:

        tok = {}

        review = input("Enter the review: ")

        if review == 'X' or review == 'x':
            exit(0)

        with open('local_file.pkl', 'rb') as handle:
            logprior, loglikelihood = pickle.load(handle)

        prob, tok = naive_bayes_predict(review, logprior, loglikelihood)

        print('Tokens: ', tok)
        print('Sentiment: ', prob)
