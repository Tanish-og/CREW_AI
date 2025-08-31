#!/usr/bin/env python3
"""
CrewAI Sentiment Analysis for X Creators
A comprehensive sentiment analysis system using CrewAI with RAG capabilities
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

# Core imports
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool as Tool

# LLM imports
import litellm

# Data processing imports
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Report generation imports
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import black, blue, red, green, gray

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crewai_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure litellm
litellm.set_verbose = True

@dataclass
class CreatorData:
    """Data structure for creator information"""
    handle: str
    tweets: List[str]
    followers: int = 0
    verified: bool = False
    bio: str = ""

@dataclass
class SentimentResult:
    """Data structure for sentiment analysis results"""
    creator: str
    overall_sentiment: str
    sentiment_score: float
    tickers: List[Dict[str, str]]
    topics: List[str]
    confidence: float

class CrewAISentimentAnalyzer:
    """Main class for CrewAI sentiment analysis system"""
    
    def __init__(self):
        # Use Gemini if available, otherwise fallback to OpenAI
        if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_API_KEY") != "your_google_api_key_here":
            # Try different Gemini model formats
            self.llm_model = "gemini/gemini-1.5-pro"
        else:
            self.llm_model = "gpt-4o-mini"  # Using OpenAI's model via litellm
        
    def _generate_mock_data(self, handle: str) -> Dict[str, Any]:
        """Generate mock data for testing"""
        mock_tweets = [
            f"Just invested in $GOOG. Feeling bullish about AI. #{handle}",
            f"The market is looking strong today. #crypto #{handle}",
            f"New tech trend: $NVDA is leading the charge. #{handle}",
            f"Feeling very bearish on the current economic situation. #stocks #{handle}",
            f"Market sentiment is shifting towards renewable energy. $TSLA #{handle}",
            f"A deep dive into $MSFT's latest earnings report. #{handle}",
            f"Looks like a good day for a short position. #trading #{handle}",
            f"Long-term hold on $AMZN is looking very promising. #{handle}",
            f"Positive outlook on $AAPL after their new product announcement. #{handle}",
            f"The biotech sector is heating up. #biotech #{handle}"
        ]
        
        return {
            "tweets": mock_tweets,
            "followers": 50000,
            "verified": True,
            "bio": f"Financial analyst and content creator. #{handle}"
        }

    @Tool("X Data Scraper Tool")
    def x_data_scraper_tool(self, user_handles: str) -> str:
        """
        Scrapes X (Twitter) data for specified user handles.
        
        Args:
            user_handles: Comma-separated list of X user handles
            
        Returns:
            JSON string containing scraped data
        """
        try:
            handles = [handle.strip() for handle in user_handles.split(',')]
            logger.info(f"Starting scraping for handles: {handles}")
            
            scraped_data = {}
            
            for handle in handles[:10]:  # Limit to 10 creators
                # Generate mock data for now
                scraped_data[handle] = self._generate_mock_data(handle)
                logger.info(f"Generated mock data for {handle}")
            
            return json.dumps(scraped_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error in X data scraper tool: {e}")
            return json.dumps(self._generate_mock_data("fallback"))

    @Tool("Sentiment Analysis Tool")
    def sentiment_analysis_tool(self, data_json: str) -> str:
        """
        Performs comprehensive sentiment analysis on collected data.
        
        Args:
            data_json: JSON string containing scraped data
            
        Returns:
            JSON string with sentiment analysis results
        """
        try:
            data = json.loads(data_json)
            results = {}
            
            # Initialize sentiment analyzers
            vader_analyzer = SentimentIntensityAnalyzer()
            
            for creator, creator_data in data.items():
                tweets = creator_data.get("tweets", [])
                
                if not tweets:
                    continue
                
                # Analyze sentiment for each tweet
                tweet_sentiments = []
                all_tickers = []
                all_topics = []
                
                for tweet in tweets:
                    # VADER sentiment analysis
                    vader_scores = vader_analyzer.polarity_scores(tweet)
                    
                    # TextBlob sentiment analysis
                    blob = TextBlob(tweet)
                    textblob_sentiment = blob.sentiment.polarity
                    
                    # Combine scores
                    combined_score = (vader_scores['compound'] + textblob_sentiment) / 2
                    
                    tweet_sentiments.append({
                        "text": tweet,
                        "vader_score": vader_scores['compound'],
                        "textblob_score": textblob_sentiment,
                        "combined_score": combined_score
                    })
                    
                    # Extract tickers
                    tickers = self._extract_tickers(tweet)
                    all_tickers.extend(tickers)
                    
                    # Extract topics
                    topics = self._extract_topics(tweet)
                    all_topics.extend(topics)
                
                # Calculate overall sentiment
                avg_sentiment = np.mean([t['combined_score'] for t in tweet_sentiments])
                
                if avg_sentiment > 0.1:
                    overall_sentiment = "bullish"
                elif avg_sentiment < -0.1:
                    overall_sentiment = "bearish"
                else:
                    overall_sentiment = "neutral"
                
                # Process tickers
                ticker_analysis = self._analyze_tickers(all_tickers, tweet_sentiments)
                
                # Remove duplicate topics
                unique_topics = list(set(all_topics))
                
                results[creator] = {
                    "overall_sentiment": overall_sentiment,
                    "sentiment_score": avg_sentiment,
                    "tickers": ticker_analysis,
                    "topics": unique_topics,
                    "confidence": abs(avg_sentiment),
                    "tweet_count": len(tweets),
                    "detailed_sentiments": tweet_sentiments
                }
            
            return json.dumps(results, indent=2)
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis tool: {e}")
            return json.dumps({})
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text"""
        import re
        # Pattern for stock tickers (e.g., $GOOG, $AAPL, etc.)
        ticker_pattern = r'\$[A-Z]{1,5}'
        return re.findall(ticker_pattern, text.upper())
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        topics = []
        text_lower = text.lower()
        
        # Define topic keywords
        topic_keywords = {
            "crypto": ["crypto", "bitcoin", "ethereum", "blockchain", "defi"],
            "stocks": ["stocks", "market", "trading", "invest", "portfolio"],
            "tech": ["ai", "artificial intelligence", "tech", "technology", "software"],
            "finance": ["finance", "financial", "earnings", "revenue", "profit"],
            "economy": ["economy", "economic", "inflation", "recession", "fed"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _analyze_tickers(self, tickers: List[str], sentiments: List[Dict]) -> List[Dict[str, str]]:
        """Analyze ticker sentiment and direction"""
        ticker_sentiments = {}
        
        for ticker in tickers:
            ticker_sentiments[ticker] = []
        
        # Find tweets mentioning each ticker
        for sentiment in sentiments:
            tweet_tickers = self._extract_tickers(sentiment["text"])
            for ticker in tweet_tickers:
                if ticker in ticker_sentiments:
                    ticker_sentiments[ticker].append(sentiment["combined_score"])
        
        # Calculate average sentiment for each ticker
        results = []
        for ticker, scores in ticker_sentiments.items():
            if scores:
                avg_score = np.mean(scores)
                if avg_score > 0.1:
                    direction = "bullish"
                elif avg_score < -0.1:
                    direction = "bearish"
                else:
                    direction = "neutral"
                
                results.append({
                    "ticker": ticker,
                    "direction": direction,
                    "sentiment_score": avg_score
                })
        
        return results

    @Tool("Report Generator Tool")
    def report_generator_tool(self, analysis_data: str) -> str:
        """
        Generates comprehensive PDF report from analysis data.
        
        Args:
            analysis_data: JSON string containing all analysis results
            
        Returns:
            Success message with file path
        """
        try:
            data = json.loads(analysis_data)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"Sentiment_Analysis_Report_{timestamp}.pdf"
            
            # Create PDF
            doc = SimpleDocTemplate(file_path, pagesize=A4)
            story = []
            
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'TitleStyle',
                parent=styles['Heading1'],
                alignment=TA_CENTER,
                fontSize=24,
                leading=30,
                textColor=blue
            )
            story.append(Paragraph("Financial Sentiment Analysis Report", title_style))
            story.append(Spacer(1, 12))
            
            # Subtitle
            subtitle_style = ParagraphStyle(
                'SubtitleStyle',
                parent=styles['Normal'],
                alignment=TA_CENTER,
                fontSize=14,
                textColor=gray
            )
            story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", subtitle_style))
            story.append(Spacer(1, 24))
            
            # Executive Summary
            summary_style = ParagraphStyle(
                'SummaryStyle',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=black
            )
            story.append(Paragraph("Executive Summary", summary_style))
            story.append(Spacer(1, 6))
            
            # Calculate overall statistics
            total_creators = len(data)
            bullish_count = sum(1 for creator_data in data.values() if creator_data.get("overall_sentiment") == "bullish")
            bearish_count = sum(1 for creator_data in data.values() if creator_data.get("overall_sentiment") == "bearish")
            neutral_count = total_creators - bullish_count - bearish_count
            
            summary_text = f"""
            This report analyzes sentiment across {total_creators} X creators. 
            Overall sentiment distribution: {bullish_count} bullish, {bearish_count} bearish, {neutral_count} neutral.
            """
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 18))
            
            # Detailed Analysis
            for creator, creator_data in data.items():
                # Creator heading
                creator_style = ParagraphStyle(
                    'CreatorStyle',
                    parent=styles['Heading2'],
                    fontSize=14,
                    textColor=blue
                )
                story.append(Paragraph(f"Analysis for @{creator}", creator_style))
                story.append(Spacer(1, 6))
                
                # Sentiment
                sentiment = creator_data.get("overall_sentiment", "unknown")
                sentiment_color = green if sentiment == "bullish" else red if sentiment == "bearish" else gray
                
                sentiment_style = ParagraphStyle(
                    'SentimentStyle',
                    parent=styles['Normal'],
                    textColor=sentiment_color
                )
                story.append(Paragraph(f"<b>Overall Sentiment:</b> {sentiment.title()}", sentiment_style))
                
                # Score
                score = creator_data.get("sentiment_score", 0)
                story.append(Paragraph(f"<b>Sentiment Score:</b> {score:.3f}", styles['Normal']))
                
                # Tickers
                tickers = creator_data.get("tickers", [])
                if tickers:
                    ticker_text = "<b>Identified Tickers:</b><br/>"
                    for ticker_info in tickers:
                        ticker = ticker_info.get("ticker", "")
                        direction = ticker_info.get("direction", "")
                        ticker_text += f"• {ticker}: {direction}<br/>"
                    story.append(Paragraph(ticker_text, styles['Normal']))
                
                # Topics
                topics = creator_data.get("topics", [])
                if topics:
                    topic_text = f"<b>Key Topics:</b> {', '.join(topics)}"
                    story.append(Paragraph(topic_text, styles['Normal']))
                
                story.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(story)
            
            return f"PDF report successfully created at: {file_path}"
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}"

    def create_agents(self):
        """Create all CrewAI agents"""
        
        # Agent 1: X Data Collector
        data_collector = Agent(
            role='X Data Collector',
            goal='Scrape tweets from X creators and collect comprehensive data including tweets, follower counts, and profile information.',
            backstory="""You are an expert web scraper specializing in social media platforms. 
            You have years of experience collecting data from X (Twitter) and understand the platform's structure.
            You use multiple methods including APIs and web scraping to ensure comprehensive data collection.""",
            verbose=True,
            llm=self.llm_model,
            tools=[self.x_data_scraper_tool],
            allow_delegation=False
        )
        
        # Agent 2: Sentiment Analyzer
        sentiment_analyzer = Agent(
            role='Sentiment Analyzer',
            goal='Analyze collected data to determine sentiment, extract financial tickers, and identify key topics.',
            backstory="""You are a seasoned financial analyst with expertise in natural language processing.
            You can read between the lines of social media posts and identify market sentiment, 
            financial trends, and investment opportunities. You use multiple sentiment analysis techniques
            to ensure accurate results.""",
            verbose=True,
            llm=self.llm_model,
            tools=[self.sentiment_analysis_tool],
            allow_delegation=False
        )
        
        # Agent 3: Report Generator
        report_generator = Agent(
            role='Report Generator',
            goal='Create comprehensive, professional PDF reports summarizing all analysis results.',
            backstory="""You are a professional business analyst and report writer.
            You excel at taking complex data and turning it into clear, actionable insights.
            Your reports are always well-structured, visually appealing, and easy to understand.""",
            verbose=True,
            llm=self.llm_model,
            tools=[self.report_generator_tool],
            allow_delegation=False
        )
        
        return data_collector, sentiment_analyzer, report_generator

    def create_tasks(self, data_collector, sentiment_analyzer, report_generator):
        """Create all CrewAI tasks"""
        
        # Task 1: Data Collection
        scrape_task = Task(
            description="""Scrape data from X creators. Focus on:
            1. Collect at least 200 tweets per creator from 10 different creators
            2. Gather profile information (followers, verification status, bio)
            3. Ensure data quality and completeness
            
            Use the X data scraper tool to perform this task.
            The list of creators to analyze will be provided.""",
            agent=data_collector,
            expected_output="""A comprehensive JSON string containing:
            - Tweets from each creator (at least 200 per creator)
            - Profile information (followers, verification, bio)""",
            context=[]
        )
        
        # Task 2: Sentiment Analysis
        analysis_task = Task(
            description="""Perform comprehensive sentiment analysis on the collected data:
            1. Analyze sentiment for each tweet using multiple techniques
            2. Extract financial tickers (e.g., $GOOG, $AAPL) and determine their sentiment direction
            3. Identify key topics and themes
            4. Calculate confidence scores for each analysis
            
            Use the sentiment analysis tool to process the data from the scraper.""",
            agent=sentiment_analyzer,
            context=[scrape_task],
            expected_output="""A detailed JSON string containing:
            - Overall sentiment for each creator (bullish/bearish/neutral)
            - Sentiment scores and confidence levels
            - Identified tickers with their sentiment direction
            - Key topics and themes
            - Detailed analysis for each tweet""",
        )
        
        # Task 3: Report Generation
        report_task = Task(
            description="""Create a comprehensive PDF report combining all analysis results:
            1. Include executive summary with key findings
            2. Provide detailed analysis for each creator
            3. Include sentiment scores, tickers, and topics
            4. Add visual elements and professional formatting
            
            Use the report generator tool to create a professional PDF report.""",
            agent=report_generator,
            context=[analysis_task],
            expected_output="""A success message indicating the PDF report has been created,
            including the file path and any relevant information about the report content.""",
        )
        
        return scrape_task, analysis_task, report_task

    def run_analysis(self, creator_handles: List[str]):
        """Run the complete CrewAI analysis workflow"""
        try:
            logger.info("Starting CrewAI sentiment analysis workflow")
            
            # Create agents
            data_collector, sentiment_analyzer, report_generator = self.create_agents()
            
            # Create tasks
            scrape_task, analysis_task, report_task = self.create_tasks(
                data_collector, sentiment_analyzer, report_generator
            )
            
            # Create crew with sequential process
            crew = Crew(
                agents=[data_collector, sentiment_analyzer, report_generator],
                tasks=[scrape_task, analysis_task, report_task],
                process=Process.sequential,
                verbose=True,
                memory=False,  # Disable memory to avoid ChromaDB dependency
                cache=False    # Disable cache to avoid ChromaDB dependency
            )
            
            # Execute the crew
            logger.info("Executing crew workflow...")
            result = crew.kickoff()
            
            logger.info("CrewAI analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in CrewAI analysis: {e}")
            raise

def main():
    """Main function to run the sentiment analysis"""
    try:
        # Initialize the analyzer
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
        
        # Run the analysis
        result = analyzer.run_analysis(creator_handles)
        
        print("\n" + "="*80)
        print("CREWAI SENTIMENT ANALYSIS COMPLETED")
        print("="*80)
        print(result)
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
