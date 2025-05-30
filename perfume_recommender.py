# Simplified Perfume Recommender System
# A real-time fragrance recommendation engine using basic Python libraries

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import re
from typing import List, Dict
import os
from collections import Counter
import math

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
            self.df = pd.read_csv(csv_path)
            print(f"Loaded {len(self.df)} perfumes from dataset")
            
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
            
        except FileNotFoundError:
            print(f"Error: Could not find {csv_path}")
            print("Please ensure your CSV file is in the same directory as this script")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces and commas
        text = re.sub(r'[^\w\s,]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
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
        
        # Create word to index mapping
        self.word_index = {word: i for i, word in enumerate(sorted(all_words))}
        print(f"Vocabulary size: {len(self.word_index)}")
    
    def create_tf_vector(self, tokens: List[str]) -> Dict[str, float]:
        """Create term frequency vector."""
        word_count = Counter(tokens)
        total_words = len(tokens)
        
        tf_vector = {}
        for word in word_count:
            if word in self.word_index:
                tf_vector[word] = word_count[word] / total_words
        
        return tf_vector
    
    def calculate_similarity(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Calculate similarity between query and document using simple cosine similarity."""
        query_tf = self.create_tf_vector(query_tokens)
        doc_tf = self.create_tf_vector(doc_tokens)
        
        # Find common words
        common_words = set(query_tf.keys()) & set(doc_tf.keys())
        
        if not common_words:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(query_tf[word] * doc_tf[word] for word in common_words)
        
        # Calculate magnitudes
        query_magnitude = math.sqrt(sum(tf ** 2 for tf in query_tf.values()))
        doc_magnitude = math.sqrt(sum(tf ** 2 for tf in doc_tf.values()))
        
        if query_magnitude == 0 or doc_magnitude == 0:
            return 0.0
        
        return dot_product / (query_magnitude * doc_magnitude)
    
    def get_recommendations(self, query: str, top_k: int = 10) -> List[Dict]:
        """Get perfume recommendations based on user query."""
        if not query.strip():
            return []
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        
        if not query_tokens:
            return []
        
        # Calculate similarities
        similarities = []
        for i, doc_tokens in enumerate(self.processed_docs):
            similarity = self.calculate_similarity(query_tokens, doc_tokens)
            similarities.append((i, similarity))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarities[:top_k]
        
        recommendations = []
        for idx, similarity_score in top_matches:
            if similarity_score > 0:  # Only include matches with some similarity
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

# Flask Web Application
app = Flask(__name__)

# Initialize recommender (global variable)
recommender = None

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint for getting recommendations."""
    global recommender
    
    if not recommender:
        return jsonify({'error': 'Recommender not initialized. Please check if perfumeDataset.csv exists.'})
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if len(query.strip()) < 2:
            return jsonify({'recommendations': []})
        
        recommendations = recommender.get_recommendations(query, top_k=8)
        return jsonify({'recommendations': recommendations})
    
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'})

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'recommender_ready': recommender is not None})

# HTML Template (same as before)
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Perfume Recommender</title>
    # <style>
    #     * {
    #         margin: 0;
    #         padding: 0;
    #         box-sizing: border-box;
    #     }
        
    #     body {
    #         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    #         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    #         min-height: 100vh;
    #         padding: 20px;
    #     }
        
    #     .container {
    #         max-width: 1200px;
    #         margin: 0 auto;
    #         background: rgba(255, 255, 255, 0.95);
    #         border-radius: 20px;
    #         padding: 30px;
    #         box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    #     }
        
    #     h1 {
    #         text-align: center;
    #         color: #333;
    #         margin-bottom: 30px;
    #         font-size: 2.5em;
    #         font-weight: 300;
    #     }
        
    #     .search-container {
    #         position: relative;
    #         margin-bottom: 30px;
    #     }
        
    #     #searchInput {
    #         width: 100%;
    #         padding: 15px 20px;
    #         font-size: 16px;
    #         border: 2px solid #ddd;
    #         border-radius: 50px;
    #         outline: none;
    #         transition: border-color 0.3s ease;
    #     }
        
    #     #searchInput:focus {
    #         border-color: #667eea;
    #     }
        
    #     .loading {
    #         display: none;
    #         text-align: center;
    #         padding: 20px;
    #         color: #666;
    #     }
        
    #     .recommendations {
    #         display: grid;
    #         grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    #         gap: 20px;
    #         margin-top: 20px;
    #     }
        
    #     .perfume-card {
    #         background: white;
    #         border-radius: 15px;
    #         padding: 20px;
    #         box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    #         transition: transform 0.3s ease, box-shadow 0.3s ease;
    #         border-left: 4px solid #667eea;
    #     }
        
    #     .perfume-card:hover {
    #         transform: translateY(-5px);
    #         box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    #     }
        
    #     .perfume-name {
    #         font-size: 1.2em;
    #         font-weight: bold;
    #         color: #333;
    #         margin-bottom: 5px;
    #     }
        
    #     .perfume-brand {
    #         color: #667eea;
    #         font-weight: 500;
    #         margin-bottom: 10px;
    #     }
        
    #     .perfume-family {
    #         background: #667eea;
    #         color: white;
    #         padding: 4px 12px;
    #         border-radius: 20px;
    #         font-size: 0.8em;
    #         display: inline-block;
    #         margin-bottom: 10px;
    #     }
        
    #     .perfume-notes {
    #         background: #f8f9fa;
    #         padding: 10px;
    #         border-radius: 8px;
    #         margin: 10px 0;
    #         font-size: 0.9em;
    #         color: #555;
    #     }
        
    #     .perfume-description {
    #         color: #666;
    #         line-height: 1.5;
    #         margin: 10px 0;
    #     }
        
    #     .perfume-price {
    #         font-weight: bold;
    #         color: #28a745;
    #         font-size: 1.1em;
    #     }
        
    #     .similarity-score {
    #         float: right;
    #         background: #28a745;
    #         color: white;
    #         padding: 2px 8px;
    #         border-radius: 10px;
    #         font-size: 0.8em;
    #     }
        
    #     .no-results {
    #         text-align: center;
    #         color: #666;
    #         font-style: italic;
    #         padding: 40px;
    #     }
        
    #     .intro-text {
    #         text-align: center;
    #         color: #666;
    #         margin-bottom: 20px;
    #         line-height: 1.6;
    #     }
        
    #     .status {
    #         text-align: center;
    #         padding: 10px;
    #         margin-bottom: 20px;
    #         border-radius: 8px;
    #     }
        
    #     .status.ready {
    #         background: #d4edda;
    #         color: #155724;
    #     }
        
    #     .status.error {
    #         background: #f8d7da;
    #         color: #721c24;
    #     }
    # </style>

    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(255, 182, 193, 0.3);
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
            font-weight: 300;
        }
        
        .search-container {
            position: relative;
            margin-bottom: 30px;
        }
        
        #searchInput {
            width: 100%;
            padding: 15px 20px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 50px;
            outline: none;
            transition: border-color 0.3s ease;
        }
        
        #searchInput:focus {
            border-color: #ff69b4;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #666;
        }
        
        .recommendations {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .perfume-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(255, 182, 193, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border-left: 4px solid #ff69b4;
        }
        
        .perfume-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(255, 105, 180, 0.25);
        }
        
        .perfume-name {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        
        .perfume-brand {
            color: #ff69b4;
            font-weight: 500;
            margin-bottom: 10px;
        }
        
        .perfume-family {
            background: linear-gradient(135deg, #ff69b4, #ff1493);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            display: inline-block;
            margin-bottom: 10px;
        }
        
        .perfume-notes {
            background: linear-gradient(135deg, #ffeef8, #ffe0f0);
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: 0.9em;
            color: #555;
            border: 1px solid #ffb6c1;
        }
        
        .perfume-description {
            color: #666;
            line-height: 1.5;
            margin: 10px 0;
        }
        
        .perfume-price {
            font-weight: bold;
            color: #e91e63;
            font-size: 1.1em;
        }
        
        .similarity-score {
            float: right;
            background: linear-gradient(135deg, #ff1493, #ff69b4);
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
        }
        
        .no-results {
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 40px;
        }
        
        .intro-text {
            text-align: center;
            color: #666;
            margin-bottom: 20px;
            line-height: 1.6;
        }
        
        .status {
            text-align: center;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 8px;
        }
        
        .status.ready {
            background: #f8e6ff;
            color: #8e24aa;
            border: 1px solid #ffb6c1;
        }
        
        .status.error {
            background: #ffe0e6;
            color: #c2185b;
            border: 1px solid #ff9eb5;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌸 Perfume Recommender</h1>
        <div id="status" class="status"></div>
        <p class="intro-text">
            Discover your perfect fragrance! Type in scent notes like "vanilla, rose, woody" 
            or describe what you're looking for like "fresh citrus for summer".
        </p>
        
        <div class="search-container">
            <input type="text" id="searchInput" placeholder="Enter fragrance notes or description (e.g., vanilla, rose, woody, fresh...)">
        </div>
        
        <div class="loading" id="loading">
            Searching for your perfect fragrance...
        </div>
        
        <div id="results" class="recommendations">
            <div class="no-results">
                Start typing to discover amazing fragrances tailored to your preferences!
            </div>
        </div>
    </div>

    <script>
        let searchTimeout;
        const searchInput = document.getElementById('searchInput');
        const resultsDiv = document.getElementById('results');
        const loadingDiv = document.getElementById('loading');
        const statusDiv = document.getElementById('status');

        // Check system status on load
        checkStatus();

        searchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            const query = this.value.trim();
            
            if (query.length < 2) {
                resultsDiv.innerHTML = '<div class="no-results">Start typing to discover amazing fragrances tailored to your preferences!</div>';
                return;
            }
            
            searchTimeout = setTimeout(() => {
                searchPerfumes(query);
            }, 300);
        });

        async function checkStatus() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                
                if (data.recommender_ready) {
                    statusDiv.innerHTML = '✅ System ready! Start searching for perfumes.';
                    statusDiv.className = 'status ready';
                } else {
                    statusDiv.innerHTML = '⚠️ System not ready. Please check if perfumeDataset.csv exists.';
                    statusDiv.className = 'status error';
                }
            } catch (error) {
                statusDiv.innerHTML = '❌ Error connecting to server.';
                statusDiv.className = 'status error';
            }
        }

        async function searchPerfumes(query) {
            loadingDiv.style.display = 'block';
            resultsDiv.innerHTML = '';
            
            try {
                const response = await fetch('/recommend', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: query })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    resultsDiv.innerHTML = `<div class="no-results">Error: ${data.error}</div>`;
                } else {
                    displayResults(data.recommendations);
                }
                
            } catch (error) {
                console.error('Error:', error);
                resultsDiv.innerHTML = '<div class="no-results">Sorry, there was an error searching for perfumes. Please try again.</div>';
            } finally {
                loadingDiv.style.display = 'none';
            }
        }

        function displayResults(recommendations) {
            if (!recommendations || recommendations.length === 0) {
                resultsDiv.innerHTML = '<div class="no-results">No perfumes found matching your search. Try different keywords!</div>';
                return;
            }
            
            const html = recommendations.map(perfume => `
                <div class="perfume-card">
                    <div class="similarity-score">${perfume.similarity_score}% match</div>
                    <div class="perfume-name">${perfume.name}</div>
                    <div class="perfume-brand">${perfume.brand}</div>
                    <div class="perfume-family">${perfume.fragrance_family}</div>
                    <div class="perfume-notes"><strong>Notes:</strong> ${perfume.notes}</div>
                    <div class="perfume-description">${perfume.description}</div>
                    ${perfume.price ? `<div class="perfume-price">$${perfume.price}</div>` : ''}
                </div>
            `).join('');
            
            resultsDiv.innerHTML = html;
        }
    </script>
</body>
</html>
'''

def setup_project():
    """Set up the project structure and files."""
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Save HTML template
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("Project setup complete!")
    print("\nFiles created:")
    print("- templates/index.html")
    print("\nTo run the application:")
    print("1. Make sure your 'perfumeDataset.csv' is in the same directory")
    print("2. Install required packages: pip install pandas flask numpy")
    print("3. Run: python perfume_recommender.py")
    print("4. Open http://localhost:5001 in your browser")

if __name__ == '__main__':
    # Setup project structure
    setup_project()
    
    # Try to initialize recommender
    if os.path.exists('perfumeDataset.csv'):
        try:
            print("Initializing perfume recommender...")
            recommender = SimplePerfumeRecommender('perfumeDataset.csv')
            print("✅ Perfume recommender initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing recommender: {e}")
    else:
        print("⚠️  perfumeDataset.csv not found. Please add your CSV file to continue.")
    
    # Run Flask app
    print("\n🚀 Starting Perfume Recommender Server...")
    print("Open http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001)
