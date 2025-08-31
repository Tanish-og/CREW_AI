#!/usr/bin/env python3
"""
Simple run script for CrewAI Sentiment Analysis
Executes the analysis with predefined creator handles
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if environment is properly set up"""
    # Check for either OpenAI or Google API key
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    # Check if keys are set and not using placeholder values
    openai_valid = openai_key and openai_key != "your_openai_api_key_here"
    google_valid = google_key and google_key != "your_google_api_key_here"
    
    if not openai_valid and not google_valid:
        print("❌ Missing required API keys")
        print("Please add either:")
        print("  - OPENAI_API_KEY (for OpenAI)")
        print("  - GOOGLE_API_KEY (for Gemini - FREE)")
        print("\nGet FREE Gemini API Key from: https://aistudio.google.com/")
        print("Or get OpenAI API Key from: https://platform.openai.com/")
        return False
    
    if google_valid:
        print("✅ Using Gemini API (FREE)")
    elif openai_valid:
        print("✅ Using OpenAI API")
    
    return True

def run_basic_analysis():
    """Run basic sentiment analysis"""
    try:
        from crewai_sentiment_analyzer import CrewAISentimentAnalyzer
        
        print("🚀 Starting CrewAI Sentiment Analysis...")
        
        # Initialize analyzer
        analyzer = CrewAISentimentAnalyzer()
        
        # Define creator handles to analyze
        creator_handles = [
            "elonmusk",
            "cz_binance", 
            "saylor",
            "VitalikButerin",
            "naval",
            "balajis",
            "michael_saylor",
            "chamath",
            "peterthiel",
            "a16z"
        ]
        
        print(f"📊 Analyzing {len(creator_handles)} creators...")
        print("Creators:", ", ".join(creator_handles))
        
        # Run the analysis
        result = analyzer.run_analysis(creator_handles)
        
        print("\n" + "="*80)
        print("✅ CREWAI SENTIMENT ANALYSIS COMPLETED")
        print("="*80)
        print(result)
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def run_advanced_analysis():
    """Run advanced analysis with RAG, YouTube, and image analysis"""
    try:
        from advanced_crewai_analyzer import AdvancedCrewAISentimentAnalyzer
        
        print("🚀 Starting Advanced CrewAI Analysis...")
        
        # Initialize advanced analyzer
        analyzer = AdvancedCrewAISentimentAnalyzer()
        
        # Create advanced agents
        rag_specialist, youtube_analyst, image_analyst = analyzer.create_advanced_agents()
        
        # Create advanced tasks
        rag_task, youtube_task, image_task = analyzer.create_advanced_tasks(
            rag_specialist, youtube_analyst, image_analyst
        )
        
        print("✅ Advanced analysis components created successfully")
        print("This includes RAG, YouTube, and Image analysis capabilities")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced analysis failed: {e}")
        return False

def main():
    """Main function"""
    print("🎯 CrewAI Sentiment Analysis Runner")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Ask user for analysis type
    print("\nChoose analysis type:")
    print("1. Basic Analysis (X scraping + sentiment + PDF report)")
    print("2. Advanced Analysis (RAG + YouTube + Image analysis)")
    print("3. Both")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
    except KeyboardInterrupt:
        print("\n👋 Analysis cancelled by user")
        sys.exit(0)
    
    success = True
    
    if choice == "1":
        success = run_basic_analysis()
    elif choice == "2":
        success = run_advanced_analysis()
    elif choice == "3":
        success = run_basic_analysis() and run_advanced_analysis()
    else:
        print("❌ Invalid choice. Please run again and select 1, 2, or 3.")
        sys.exit(1)
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        print("📄 Check the generated PDF report for detailed results")
        print("📊 Check the logs for detailed processing information")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
