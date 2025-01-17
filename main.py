from src.analyzer import ChatbotLogAnalyzer
from src.visualizer import ChatbotVisualizer
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """
    Main function to run the chatbot log analysis.
    """
    try:
        # Initialize analyzer
        logging.info("Starting chatbot log analysis...")
        analyzer = ChatbotLogAnalyzer('chatbot_logs.csv')
        
        # Generate and display basic statistics
        stats = analyzer.get_basic_stats()
        logging.info("\nBasic Statistics:")
        for key, value in stats.items():
            logging.info(f"{key}: {value}")
        
        # Analyze time patterns
        time_patterns = analyzer.analyze_time_patterns()
        logging.info("\nHourly Usage Patterns:")
        logging.info(time_patterns)
        
        # Analyze errors
        error_counts, error_impact = analyzer.analyze_errors()
        logging.info("\nError Distribution:")
        logging.info(error_counts)
        logging.info("\nError Impact on Performance:")
        logging.info(error_impact)
        
        # Create visualizations
        visualizer = ChatbotVisualizer()
        viz_data = analyzer.get_data_for_visualization()
        
        # Create output directory if it doesn't exist
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        # Generate and save visualizations
        visualizer.create_dashboard(viz_data, 'output/chatbot_analysis_dashboard.png')
        logging.info("Analysis complete. Visualizations saved to 'output' directory.")
        
    except Exception as e:
        logging.error(f"An error occurred during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()