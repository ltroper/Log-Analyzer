import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

class ChatbotVisualizer:
    """
    Creates visualizations for chatbot log analysis results.
    """
    
    def __init__(self):
        """Initialize the visualizer with default style settings."""
        sns.set_theme()
        self.fig_size = (15, 12)
        self.colors = sns.color_palette("husl", 8)
    
    def create_dashboard(self, data: Dict, save_path: str = None) -> None:
        """
        Generate a comprehensive dashboard of visualizations.
        
        Args:
            data (Dict): Dictionary containing processed data for plotting
            save_path (str, optional): Path to save the visualization. Defaults to None.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.fig_size)
        
        # 1. Hourly interaction pattern
        sns.barplot(x=data['hourly_interactions'].index, 
                   y=data['hourly_interactions'].values, 
                   ax=axes[0,0],
                   color=self.colors[0])
        axes[0,0].set_title('Interactions by Hour')
        axes[0,0].set_xlabel('Hour of Day')
        axes[0,0].set_ylabel('Number of Interactions')
        
        # 2. Latency distribution
        sns.histplot(data=data['latency_distribution'], 
                    bins=30, 
                    ax=axes[0,1],
                    color=self.colors[1])
        axes[0,1].set_title('Latency Distribution')
        axes[0,1].set_xlabel('Latency (ms)')
        
        # 3. Language distribution
        lang_data = data['language_distribution']
        axes[1,0].pie(lang_data.values, 
                     labels=lang_data.index, 
                     autopct='%1.1f%%',
                     colors=[self.colors[2], self.colors[3]])
        axes[1,0].set_title('Language Distribution')
        
        # 4. Token usage over time
        sns.scatterplot(data=data['token_usage'], 
                       x='date', 
                       y='total_tokens', 
                       alpha=0.5, 
                       ax=axes[1,1],
                       color=self.colors[4])
        axes[1,1].set_title('Token Usage Over Time')
        axes[1,1].set_xlabel('Date')
        axes[1,1].set_ylabel('Total Tokens')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()