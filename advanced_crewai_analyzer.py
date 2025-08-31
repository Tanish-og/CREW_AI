#!/usr/bin/env python3
"""
Advanced CrewAI Sentiment Analysis with RAG, YouTube, and Image Analysis
Extended version with additional capabilities
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
import requests

# Core imports
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool as Tool

# LLM and RAG imports
import litellm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# YouTube analysis imports
from youtube_transcript_api import YouTubeTranscriptApi

# Image processing imports
from PIL import Image
import torch
from transformers import pipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_crewai_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedCrewAISentimentAnalyzer:
    """Advanced CrewAI sentiment analysis with RAG, YouTube, and image analysis"""
    
    def __init__(self):
        self.llm_model = "gpt-4o-mini"
        self.vector_store = None
        self.embeddings = None
        self.image_analyzer = None
        self.setup_components()
        
    def setup_components(self):
        """Initialize advanced components"""
        try:
            # Setup embeddings for RAG
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Setup image analysis pipeline
            self.image_analyzer = pipeline(
                "image-classification",
                model="microsoft/resnet-50"
            )
            
            logger.info("Advanced components initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up advanced components: {e}")
    
    def setup_rag(self, documents: List[str]):
        """Setup RAG with documents"""
        try:
            if not self.embeddings:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            
            # Create vector store
            self.vector_store = Chroma.from_texts(
                documents,
                self.embeddings,
                persist_directory="./chroma_db"
            )
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up RAG: {e}")

    @Tool("YouTube Data Collector Tool")
    def youtube_data_collector_tool(self, creator_handles: str) -> str:
        """
        Collects YouTube data for creators to enhance RAG analysis.
        
        Args:
            creator_handles: Comma-separated list of creator handles
            
        Returns:
            JSON string containing YouTube data
        """
        try:
            handles = [handle.strip() for handle in creator_handles.split(',')]
            youtube_data = {}
            
            for handle in handles[:5]:  # Limit to 5 creators for YouTube
                try:
                    # Search for YouTube channels
                    search_url = f"https://www.googleapis.com/youtube/v3/search"
                    params = {
                        "part": "snippet",
                        "q": handle,
                        "type": "channel",
                        "key": os.getenv("YOUTUBE_API_KEY"),
                        "maxResults": 1
                    }
                    
                    response = requests.get(search_url, params=params)
                    response.raise_for_status()
                    
                    data = response.json()
                    if data.get("items"):
                        channel_id = data["items"][0]["id"]["channelId"]
                        
                        # Get recent videos
                        videos_url = f"https://www.googleapis.com/youtube/v3/search"
                        video_params = {
                            "part": "snippet",
                            "channelId": channel_id,
                            "order": "date",
                            "type": "video",
                            "key": os.getenv("YOUTUBE_API_KEY"),
                            "maxResults": 10
                        }
                        
                        video_response = requests.get(videos_url, params=video_params)
                        video_response.raise_for_status()
                        
                        video_data = video_response.json()
                        videos = []
                        
                        for video in video_data.get("items", []):
                            video_id = video["id"]["videoId"]
                            
                            # Get transcript if available
                            try:
                                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                                transcript_text = " ".join([t["text"] for t in transcript])
                                videos.append({
                                    "title": video["snippet"]["title"],
                                    "transcript": transcript_text,
                                    "video_id": video_id
                                })
                            except:
                                videos.append({
                                    "title": video["snippet"]["title"],
                                    "transcript": "",
                                    "video_id": video_id
                                })
                        
                        youtube_data[handle] = {
                            "channel_id": channel_id,
                            "videos": videos
                        }
                
                except Exception as e:
                    logger.error(f"Error collecting YouTube data for {handle}: {e}")
            
            return json.dumps(youtube_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error in YouTube data collector tool: {e}")
            return json.dumps({})

    @Tool("RAG Analysis Tool")
    def rag_analysis_tool(self, data_json: str) -> str:
        """
        Performs RAG-based analysis using vector search.
        
        Args:
            data_json: JSON string containing data to analyze
            
        Returns:
            JSON string with RAG analysis results
        """
        try:
            data = json.loads(data_json)
            
            if not self.vector_store:
                # Setup RAG with the data
                documents = []
                for creator, creator_data in data.items():
                    tweets = creator_data.get("tweets", [])
                    documents.extend(tweets)
                
                self.setup_rag(documents)
            
            # Perform similarity search for insights
            insights = {}
            
            for creator, creator_data in data.items():
                tweets = creator_data.get("tweets", [])
                
                if tweets and self.vector_store:
                    # Find similar content across all creators
                    similar_docs = self.vector_store.similarity_search(
                        " ".join(tweets[:5]),  # Use first 5 tweets as query
                        k=10
                    )
                    
                    # Extract insights from similar documents
                    related_topics = []
                    for doc in similar_docs:
                        doc_text = doc.page_content
                        # Extract topics from document text
                        topics = self._extract_topics(doc_text)
                        related_topics.extend(topics)
                    
                    insights[creator] = {
                        "related_topics": list(set(related_topics)),
                        "similarity_analysis": "Based on vector similarity search"
                    }
            
            return json.dumps(insights, indent=2)
            
        except Exception as e:
            logger.error(f"Error in RAG analysis tool: {e}")
            return json.dumps({})

    @Tool("Image Analysis Tool")
    def image_analysis_tool(self, image_urls: str) -> str:
        """
        Analyzes images for relevant financial content.
        
        Args:
            image_urls: JSON string containing image URLs to analyze
            
        Returns:
            JSON string with image analysis results
        """
        try:
            urls = json.loads(image_urls)
            results = []
            
            for url in urls:
                try:
                    # Download and analyze image
                    response = requests.get(url)
                    response.raise_for_status()
                    
                    # Save image temporarily
                    with open("temp_image.jpg", "wb") as f:
                        f.write(response.content)
                    
                    # Analyze image
                    if self.image_analyzer:
                        image = Image.open("temp_image.jpg")
                        predictions = self.image_analyzer(image)
                        
                        # Check if image is relevant to finance
                        relevant_keywords = ["chart", "graph", "money", "business", "finance"]
                        is_relevant = any(
                            keyword in pred["label"].lower() 
                            for pred in predictions[:3] 
                            for keyword in relevant_keywords
                        )
                        
                        results.append({
                            "url": url,
                            "is_relevant": is_relevant,
                            "predictions": predictions[:5],
                            "confidence": max([p["score"] for p in predictions[:3]])
                        })
                    
                    # Clean up
                    os.remove("temp_image.jpg")
                    
                except Exception as e:
                    logger.error(f"Error analyzing image {url}: {e}")
                    results.append({
                        "url": url,
                        "error": str(e)
                    })
            
            return json.dumps(results, indent=2)
            
        except Exception as e:
            logger.error(f"Error in image analysis tool: {e}")
            return json.dumps([])

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

    def create_advanced_agents(self):
        """Create advanced CrewAI agents"""
        
        # Agent 1: RAG Specialist
        rag_specialist = Agent(
            role='RAG Specialist',
            goal='Perform advanced analysis using RAG (Retrieval-Augmented Generation) to find patterns and insights across creators.',
            backstory="""You are a data scientist specializing in vector databases and semantic search.
            You can identify patterns across large datasets and find hidden connections between
            different creators and their content. You use advanced NLP techniques to extract insights.""",
            verbose=True,
            llm=self.llm_model,
            tools=[self.rag_analysis_tool],
            allow_delegation=False
        )
        
        # Agent 2: YouTube Analyst
        youtube_analyst = Agent(
            role='YouTube Analyst',
            goal='Analyze YouTube content from creators to enhance sentiment analysis with video transcripts and content.',
            backstory="""You are a content analyst specializing in video content analysis.
            You can extract insights from video transcripts and identify patterns in video content
            that complement social media sentiment analysis.""",
            verbose=True,
            llm=self.llm_model,
            tools=[self.youtube_data_collector_tool],
            allow_delegation=False
        )
        
        # Agent 3: Image Analyst
        image_analyst = Agent(
            role='Image Analyst',
            goal='Analyze images for relevant financial content and identify charts, graphs, and other visual data.',
            backstory="""You are a computer vision expert specializing in financial imagery analysis.
            You can identify charts, graphs, financial documents, and other visual content that
            might contain important financial information. You use state-of-the-art image recognition models.""",
            verbose=True,
            llm=self.llm_model,
            tools=[self.image_analysis_tool],
            allow_delegation=False
        )
        
        return rag_specialist, youtube_analyst, image_analyst

    def create_advanced_tasks(self, rag_specialist, youtube_analyst, image_analyst):
        """Create advanced CrewAI tasks"""
        
        # Task 1: RAG Analysis
        rag_task = Task(
            description="""Perform RAG-based analysis to find patterns and insights:
            1. Use vector similarity search to find related content across creators
            2. Identify common themes and patterns
            3. Find hidden connections between different creators
            4. Provide additional context and insights
            
            Use the RAG analysis tool to enhance the sentiment analysis results.""",
            agent=rag_specialist,
            expected_output="""A JSON string containing:
            - Related topics and themes across creators
            - Similarity analysis results
            - Additional insights from vector search
            - Pattern identification across the dataset""",
        )
        
        # Task 2: YouTube Analysis
        youtube_task = Task(
            description="""Analyze YouTube content from creators:
            1. Collect video transcripts and metadata
            2. Identify financial topics and sentiment in video content
            3. Compare video sentiment with social media sentiment
            4. Extract key insights from video content
            
            Use the YouTube data collector tool to gather and analyze video content.""",
            agent=youtube_analyst,
            expected_output="""A JSON string containing:
            - Video transcripts and metadata
            - Financial topics identified in videos
            - Sentiment analysis of video content
            - Comparison with social media sentiment""",
        )
        
        # Task 3: Image Analysis
        image_task = Task(
            description="""Analyze any images or visual content for financial relevance:
            1. Identify charts, graphs, and financial documents
            2. Determine if images contain relevant financial information
            3. Extract any text or data from images
            4. Assess the importance of visual content
            
            Use the image analysis tool to process any image URLs found in the data.""",
            agent=image_analyst,
            expected_output="""A JSON string containing:
            - Analysis results for each image
            - Relevance assessment for financial content
            - Confidence scores for image classification
            - Any extracted text or data from images""",
        )
        
        return rag_task, youtube_task, image_task

def main():
    """Main function to run advanced analysis"""
    try:
        # Initialize the advanced analyzer
        analyzer = AdvancedCrewAISentimentAnalyzer()
        
        # Create advanced agents
        rag_specialist, youtube_analyst, image_analyst = analyzer.create_advanced_agents()
        
        # Create advanced tasks
        rag_task, youtube_task, image_task = analyzer.create_advanced_tasks(
            rag_specialist, youtube_analyst, image_analyst
        )
        
        print("Advanced CrewAI components created successfully")
        print("This module provides RAG, YouTube, and Image analysis capabilities")
        
    except Exception as e:
        logger.error(f"Error in advanced analysis: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
