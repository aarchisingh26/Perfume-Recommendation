import pandas as pd
import numpy as np
import re
from collections import Counter
import math
import json
import os
from urllib.parse import parse_qs

class SimplePerfumeRecommender:
    def __init__(self, csv_path: str = 'perfumeDataset.csv'):
        """Initialize the perfume recommender system."""
        self.df = None
        self.word_index = {}
        self.processed_docs = []
        self.load_and_preprocess_data(csv_path)
        
    def load_and_preprocess_data(self, csv_path: str):
        """Load and preprocess the perfume dataset."""
        try:
            # Try to find the CSV file in the current directory or parent
            if not os.path.exists(csv_path):
                csv_path = os.path.join('..', csv_path)
            
            self.df = pd.read_csv(csv_path)
            
            # Handle missing values
            self.df['description'] = self.df['description'].fillna('')
            self.df['notes'] = self.df['notes'].fillna('')
            self.df['fragranceFamily'] = self.df['fragranceFamily'].fillna('')
            self.df['name'] = self.df['name'].fillna('Unknown')
            self.df['brand'] = self.df['brand'].fillna('Unknown')
            self.df['price_50ml'] = self.df['price_50ml'].fillna(0)
            
            # Create combined features for analysis
            self.df['combined_features'] = (
                self.df['description'].astype(str) + ' ' +
                self.df['notes'].astype(str) + ' ' +
                self.df['fragranceFamily'].astype(str)
            )
            
            # Clean and process text
            self.process_documents()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if pd.isna(text):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^\w\s,]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text: str) -> list:
        """Simple tokenization and stop word removal."""
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'do', 'how', 'their', 'if',
            'up', 'out', 'many', 'then', 'them', 'so', 'some', 'her', 'would',
            'make', 'like', 'into', 'him', 'time', 'two', 'more', 'go', 'no',
            'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'oil',
            'water', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part'
        }
        
        words = self.clean_text(text).split()
        return [word for word in words if word and len(word) > 2 and word not in stop_words]
    
    def process_documents(self):
        """Process all documents and create word index."""
        all_words = set()
        self.processed_docs = []
        
        for combined_text in self.df['combined_features']:
            tokens = self.tokenize(combined_text)
            self.processed_docs.append(tokens)
            all_words.update(tokens)
        
        self.word_index = {word: i for i, word in enumerate(sorted(all_words))}
    
    def create_tf_vector(self, tokens: list) -> dict:
        """Create term frequency vector."""
        word_count = Counter(tokens)
        total_words = len(tokens)
        
        tf_vector = {}
        for word in word_count:
            if word in self.word_index:
                tf_vector[word] = word_count[word] / total_words
        
        return tf_vector
    
    def calculate_similarity(self, query_tokens: list, doc_tokens: list) -> float:
        """Calculate similarity between query and document."""
        query_tf = self.create_tf_vector(query_tokens)
        doc_tf = self.create_tf_vector(doc_tokens)
        
        common_words = set(query_tf.keys()) & set(doc_tf.keys())
        
        if not common_words:
            return 0.0
        
        dot_product = sum(query_tf[word] * doc_tf[word] for word in common_words)
        
        query_magnitude = math.sqrt(sum(tf ** 2 for tf in query_tf.values()))
        doc_magnitude = math.sqrt(sum(tf ** 2 for tf in doc_tf.values()))
        
        if query_magnitude == 0 or doc_magnitude == 0:
            return 0.0
        
        return dot_product / (query_magnitude * doc_magnitude)
    
    def get_recommendations(self, query: str, top_k: int = 10) -> list:
        """Get perfume recommendations based on user query."""
        if not query.strip():
            return []
        
        query_tokens = self.tokenize(query)
        
        if not query_tokens:
            return []
        
        similarities = []
        for i, doc_tokens in enumerate(self.processed_docs):
            similarity = self.calculate_similarity(query_tokens, doc_tokens)
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarities[:top_k]
        
        recommendations = []
        for idx, similarity_score in top_matches:
            if similarity_score > 0:
                rec = {
                    'name': str(self.df.iloc[idx]['name']),
                    'brand': str(self.df.iloc[idx]['brand']),
                    'description': str(self.df.iloc[idx]['description'])[:200] + ('...' if len(str(self.df.iloc[idx]['description'])) > 200 else ''),
                    'notes': str(self.df.iloc[idx]['notes']),
                    'fragrance_family': str(self.df.iloc[idx]['fragranceFamily']),
                    'price': float(self.df.iloc[idx]['price_50ml']) if pd.notna(self.df.iloc[idx]['price_50ml']) else 0,
                    'similarity_score': round(similarity_score * 100, 2)
                }
                recommendations.append(rec)
        
        return recommendations

# Initialize recommender globally (cached across requests)
recommender = None

def init_recommender():
    global recommender
    if recommender is None:
        try:
            recommender = SimplePerfumeRecommender()
        except Exception as e:
            print(f"Failed to initialize recommender: {e}")
            return False
    return True

def handler(request):
    """Vercel serverless function handler."""
    # Set CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json'
    }
    
    # Handle OPTIONS request for CORS
    if request.method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    # Initialize recommender
    if not init_recommender():
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': 'Failed to initialize recommender system'})
        }
    
    try:
        if request.method == 'POST':
            # Parse request body
            body = request.get_json() if hasattr(request, 'get_json') else json.loads(request.body or '{}')
            query = body.get('query', '')
            
            if len(query.strip()) < 2:
                recommendations = []
            else:
                recommendations = recommender.get_recommendations(query, top_k=8)
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({'recommendations': recommendations})
            }
        else:
            return {
                'statusCode': 405,
                'headers': headers,
                'body': json.dumps({'error': 'Method not allowed'})
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': f'Server error: {str(e)}'})
        }
