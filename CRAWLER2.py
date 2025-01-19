import requests
from bs4 import BeautifulSoup
import json
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import nltk

# Εγκατάσταση του ntlk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text.lower())

    # Stop-word removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

def crawl_wikipedia(query, max_results=5):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": max_results
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        search_results = response.json().get('query', {}).get('search', [])
        articles = []
        
        for result in search_results:
            title = result['title']
            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            articles.append(extract_wikipedia_article(page_url))
        
        return articles
    else:
        print(f"Failed to fetch search results for query: {query}")
        return []

def extract_wikipedia_article(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.find('h1', {'id': 'firstHeading'}).text.strip()
        content = ''
        for paragraph in soup.find_all('p'):
            content += paragraph.text.strip() + ' '

        
        preprocessed_title = preprocess_text(title)
        preprocessed_content = preprocess_text(content)

        print(f"Title: {title}, Preprocessed Title: {preprocessed_title}")
        print(f"Content Sample: {content[:100]}, Preprocessed Content: {preprocessed_content[:10]}")

        article = {
            'url': url,
            'title': title,
            'content': content,
            'preprocessed_title': preprocessed_title,
            'preprocessed_content': preprocessed_content
        }

        return article
    else:
        print(f"Failed to fetch article from {url}")
        return None


def create_and_save_reverse_index(articles):
    reverse_index = defaultdict(list)

    for idx, article in enumerate(articles):
        for term in set(article['preprocessed_title'] + article['preprocessed_content']):
            reverse_index[term].append(idx)

    print(f"Reverse Index Sample: {dict(list(reverse_index.items())[:10])}")
    save_reverse_index(reverse_index)

   

def save_to_json(articles, filename='wikipedia_articles.json'):
    with open(filename, 'w') as json_file:
        json.dump(articles, json_file, indent=2)
    print(f'Saved {len(articles)} articles to {filename}')

def save_reverse_index(reverse_index, filename='reverse_index.json'):
    with open(filename, 'w') as json_file:
        json.dump(reverse_index, json_file, indent=2)
    print(f'Saved reverse index to {filename}')
    
if __name__ == '__main__':
    query = input("Enter your search query for Wikipedia: ")
    wikipedia_articles = crawl_wikipedia(query)
    save_to_json(wikipedia_articles)

    # Δημιουργια και αποθήκευση του αρχείου
    create_and_save_reverse_index(wikipedia_articles)
