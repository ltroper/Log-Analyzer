import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

class ChatbotLogAnalyzer:
    """
    Analyzes chatbot interaction logs to extract meaningful insights about usage patterns,
    performance metrics, and user behavior.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize the analyzer with the path to the CSV file containing chatbot logs.
        
        Args:
            csv_path (str): Path to the CSV file with chatbot logs
        """
        self.df = pd.read_csv(csv_path)
        self._preprocess_data()
    
    def _preprocess_data(self) -> None:
        """
        Prepare the data for analysis by converting data types and creating derived features.
        """
        # Convert date to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Extract useful time components
        self.df['hour'] = self.df['date'].dt.hour
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        
        # Convert latency to milliseconds for better readability
        self.df['latency_ms'] = self.df['latency'] * 1000
        
        # Create binary columns for error analysis
        self.df['has_error'] = self.df['error'].ne('NONE')
        
        # Create language column based on outputs
        self.df['language'] = self.df['outputs'].apply(
            lambda x: 'Spanish' if any(word in str(x).lower() for word in 
                ['hola', 'gracias', 'buenos días']) else 'English'
        )

    def get_basic_stats(self) -> Dict:
        """
        Calculate basic usage statistics from the chatbot logs.
        
        Returns:
            Dict: Dictionary containing key metrics about chatbot usage
        """
        stats = {
            'total_interactions': len(self.df),
            'unique_users': self.df['user_id'].nunique(),
            'success_rate': (self.df['is_flow_successful'].mean() * 100).round(2),
            'avg_latency_ms': self.df['latency_ms'].mean().round(2),
            'avg_tokens': self.df['total_tokens'].mean().round(2),
            'languages': self.df['language'].value_counts().to_dict()
        }
        return stats

    def analyze_time_patterns(self) -> pd.DataFrame:
        """
        Analyze usage patterns by hour of day.
        
        Returns:
            pd.DataFrame: Hourly statistics including interaction counts and performance metrics
        """
        hourly_stats = self.df.groupby('hour').agg({
            'user_id': 'count',
            'latency_ms': 'mean',
            'total_tokens': 'mean',
            'has_error': 'mean'
        }).round(2)
        
        hourly_stats.columns = ['Interactions', 'Avg Latency (ms)', 
                              'Avg Tokens', 'Error Rate']
        return hourly_stats

    def analyze_errors(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Analyze error patterns and their impact on performance.
        
        Returns:
            Tuple[pd.Series, pd.DataFrame]: Error counts and impact analysis
        """
        error_counts = self.df['error'].value_counts()
        
        error_impact = self.df.groupby('has_error').agg({
            'latency_ms': ['mean', 'std'],
            'total_tokens': ['mean', 'std']
        }).round(2)
        
        return error_counts, error_impact

    def get_data_for_visualization(self) -> Dict:
        """
        Prepare data for visualization functions.
        
        Returns:
            Dict: Dictionary containing processed data ready for plotting
        """
        return {
            'hourly_interactions': self.df.groupby('hour')['user_id'].count(),
            'latency_distribution': self.df['latency_ms'],
            'language_distribution': self.df['language'].value_counts(),
            'token_usage': self.df[['date', 'total_tokens']],
            'daily_interactions': self.df.groupby('day_of_week')['user_id'].count()
        }