import json
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from math import log
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import nltk


# Εγκατάσταση nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def evaluate_system(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    map_score = average_precision_score(true_labels, predicted_labels)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Mean Average Precision (MAP): {map_score:.4f}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map": map_score
    }

def load_articles(filename='wikipedia_articles.json'):
    with open(filename, 'r') as json_file:
        articles = json.load(json_file)
    return articles

def load_reverse_index(filename='reverse_index.json'):
    with open(filename, 'r') as json_file:
        reverse_index = json.load(json_file)
    return reverse_index

def boolean_query_processing(query, reverse_index):
    """
    Διαχειρίζεται ερωτήματα Boolean που περιέχουν AND, OR, NOT.
    """
    query = query.upper()  # Ensure operators are uppercase
    query_tokens = re.findall(r'[\w]+|AND|OR|NOT|\(|\)', query)  # Tokenize the query

  
    term_postings = {}
    for token in query_tokens:
        if token not in {'AND', 'OR', 'NOT', '(', ')'}:
            preprocessed_token = preprocess_text(token)[0]  # Single word preprocessing
            if preprocessed_token in reverse_index:
                term_postings[token] = set(reverse_index[preprocessed_token])
            else:
                term_postings[token] = set()

    # AND OR NOT BOOLEAN
    def eval_query(tokens):
        stack = []
        for token in tokens:
            if token == "AND":
                b = stack.pop()
                a = stack.pop()
                stack.append(a & b)
            elif token == "OR":
                b = stack.pop()
                a = stack.pop()
                stack.append(a | b)
            elif token == "NOT":
                a = stack.pop()
                stack.append(set(range(len(reverse_index))) - a)
            else:
                stack.append(term_postings.get(token, set()))
        return stack[0]

    # 
    def infix_to_postfix(tokens):
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
        output = []
        operators = []

        for token in tokens:
            if token not in precedence and token not in {'(', ')'}:
                output.append(token)
            elif token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()  # Remove '('
            else:
                while (operators and operators[-1] != '(' and
                       precedence.get(token, 0) <= precedence.get(operators[-1], 0)):
                    output.append(operators.pop())
                operators.append(token)

        while operators:
            output.append(operators.pop())

        return output

    postfix_query = infix_to_postfix(query_tokens)
    return list(eval_query(postfix_query))


def vector_space_model(query, reverse_index, articles):
    query_tokens = preprocess_text(query)
    query_vector = defaultdict(int)
    result_scores = defaultdict(float)

    for term in query_tokens:
        query_vector[term] += 1

    for term, query_tf in query_vector.items():
        if term in reverse_index:
            idf = log(len(articles) / len(reverse_index[term]))
            for doc_id in reverse_index[term]:
                tf = articles[doc_id]['preprocessed_title'].count(term) + articles[doc_id]['preprocessed_content'].count(term)
                result_scores[doc_id] += tf * idf * query_tf

    result_docs = sorted(result_scores.keys(), key=lambda doc_id: result_scores[doc_id], reverse=True)
    return result_docs

def okapi_bm25(query, reverse_index, articles, k1=1.5, b=0.75):
    query_tokens = preprocess_text(query)
    result_scores = defaultdict(float)

    avg_doc_len = sum(len(article['preprocessed_title']) + len(article['preprocessed_content']) for article in articles) / len(articles)

    for term in query_tokens:
        if term in reverse_index:
            df = len(reverse_index[term])
            idf = log((len(articles) - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id in reverse_index[term]:
                tf = articles[doc_id]['preprocessed_title'].count(term) + articles[doc_id]['preprocessed_content'].count(term)
                doc_len = len(articles[doc_id]['preprocessed_title']) + len(articles[doc_id]['preprocessed_content'])
                bm25_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                result_scores[doc_id] += bm25_score

    result_docs = sorted(result_scores.keys(), key=lambda doc_id: result_scores[doc_id], reverse=True)
    return result_docs

def rank(result_doc_ids, query, articles):
    query_tokens = preprocess_text(query)
    ranked_results = []

    for doc_id in result_doc_ids:
        score = 0
        for term in query_tokens:
            score += articles[doc_id]['preprocessed_title'].count(term) + articles[doc_id]['preprocessed_content'].count(term)
        ranked_results.append((doc_id, score))
        
    ranked_results = sorted(ranked_results, key=lambda result: result[1], reverse=True)
    return [result[0] for result in ranked_results]

def evaluate_queries(test_queries, articles, reverse_index, algo="BOOLEAN"):
    all_metrics = []

    for test_query in test_queries:
        query = test_query["query"]
        relevant_docs = set(test_query["relevant_docs"])

        if algo == "BOOLEAN":
            result_docs = boolean_query_processing(query, reverse_index)
        elif algo == "VSM":
            result_docs = vector_space_model(query, reverse_index, articles)
        elif algo == "OBM":
            result_docs = okapi_bm25(query, reverse_index, articles)



        # Υπολογισμός True και Predicted Labels
        true_labels = [1 if i in relevant_docs else 0 for i in range(len(articles))]
        predicted_labels = [1 if i in result_docs else 0 for i in range(len(articles))]

        print(f"Query: {query}")
        print(f"Relevant Docs: {relevant_docs}")
        print(f"Retrieved Docs: {set(result_docs)}")
        print(f"True Labels: {true_labels}")
        print(f"Predicted Labels: {predicted_labels}")

        # Υπολογισμός Μετρικών
        metrics = evaluate_system(true_labels, predicted_labels)
        all_metrics.append(metrics)

    return all_metrics

if __name__ == '__main__':
    articles = load_articles()
    reverse_index = load_reverse_index()

    test_queries = [
        {"query": "machine learning", "relevant_docs": [0, 2]},
        {"query": "artificial intelligence", "relevant_docs": [1, 3]},
        {"query": "data science", "relevant_docs": [2]},
        {"query": "deep learning", "relevant_docs": [0, 1]},
        {"query": "information retrieval", "relevant_docs": [3]}
    ]

    algo = input("Choose algorithm (BOOLEAN, VSM, OBM): ").strip().upper()
    if algo not in {"BOOLEAN", "VSM", "OBM"}:
        print("Invalid algorithm choice. Exiting...")
    else:
        evaluate_queries(test_queries, articles, reverse_index, algo=algo)
