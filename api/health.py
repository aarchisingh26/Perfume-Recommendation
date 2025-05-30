import json
import os

def handler(request):
    """Health check endpoint."""
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json'
    }
    
    if request.method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    # Check if CSV file exists
    csv_exists = os.path.exists('perfumeDataset.csv') or os.path.exists('../perfumeDataset.csv')
    
    return {
        'statusCode': 200,
        'headers': headers,
        'body': json.dumps({
            'status': 'healthy',
            'recommender_ready': csv_exists
        })
    }