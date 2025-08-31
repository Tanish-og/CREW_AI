# main.py

import os
import json
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool as Tool
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER

# --- NEW DEBUGGING CODE ---
def check_env_file():
    """Checks if the .env file exists and contains the API key."""
    env_path = os.path.join(os.getcwd(), '.env')
    if not os.path.exists(env_path):
        print(f"Error: The .env file was not found at {env_path}")
        return False
    
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: The GOOGLE_API_KEY was not found in the .env file. Please check its contents.")
        return False
        
    print("Success: .env file found and API key loaded.")
    return True

if not check_env_file():
    exit()
# --- END NEW DEBUGGING CODE ---

# ==============================================================================
# 1. SET UP THE LLM (Large Language Model)
# ==============================================================================
# The litellm library automatically picks up the 'GOOGLE_API_KEY'
# from your environment variables.
# For this assessment, we'll use a Gemini model.
llm_model = "gemini/gemini-2.5-flash-preview"

# ==============================================================================
# 2. DEFINE YOUR CUSTOM TOOLS
# ==============================================================================
# Tools are functions that your agents can use to perform actions.
# You must implement the actual logic for these.

@Tool("X Data Scraper Tool")
def x_data_scraper_tool(user_handles: str) -> str:
    """
    Scrapes a list of X (formerly Twitter) user handles for their tweets
    and returns a JSON-formatted string of the collected data.
    
    Args:
        user_handles: A comma-separated string of X user handles to scrape.
    
    Returns:
        A JSON string containing the creator and a list of their tweets.
        
    NOTE: You must replace the placeholder logic with an actual scraping solution.
    A free solution might use a library like `twikit`. A paid solution would
    be an API like Brightdata as mentioned in the prompt.
    """
    print(f"Executing scraping tool for users: {user_handles}...")
    
    # Placeholder for the scraped data. You must replace this with
    # the actual output from your scraping tool.
    
    # In a real scenario, you would loop through user_handles and scrape each one.
    
    mock_data = {
        "CreatorA": {
            "tweets": [
                "Just invested in $GOOG. Feeling bullish about AI.",
                "The market is looking strong today. #crypto",
                "New tech trend: $NVDA is leading the charge.",
                "Feeling very bearish on the current economic situation. #stocks"
            ]
        },
        "CreatorB": {
            "tweets": [
                "Market sentiment is shifting towards renewable energy. $TSLA",
                "A deep dive into $MSFT's latest earnings report.",
                "Looks like a good day for a short position. #trading",
                "Long-term hold on $AMZN is looking very promising."
            ]
        },
        "CreatorC": {
            "tweets": [
                "Positive outlook on $AAPL after their new product announcement.",
                "The biotech sector is heating up. #biotech",
                "Thinking about a new strategy for my portfolio.",
                "The market is unpredictable, but analysis helps. #finance"
            ]
        },
        "CreatorD": {
            "tweets": [
                "Another day, another rally. $SPY is up.",
                "Looking at potential opportunities in the gaming industry. $SONY",
                "The Q2 reports are out, and it's time to analyze.",
                "Bearish sentiment is high, but I'm staying positive."
            ]
        },
        "CreatorE": {
            "tweets": [
                "Excited about the future of AI and robotics. $IRBT",
                "A breakdown of the recent crypto market volatility.",
                "Shorting is a high-risk game.",
                "The future is now. $GOOG"
            ]
        }
    }
    
    return json.dumps(mock_data)

@Tool("PDF Generator Tool")
def pdf_generator_tool(report_data: str) -> str:
    """
    Generates a professional PDF report from the provided JSON data.
    
    Args:
        report_data: A JSON string containing the final structured report data.
    
    Returns:
        A success message indicating the PDF file has been created.
    """
    data = json.loads(report_data)
    
    file_path = "Sentiment_Report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        alignment=TA_CENTER,
        fontSize=24,
        leading=30
    )
    story.append(Paragraph("Financial Sentiment Analysis Report", title_style))
    story.append(Spacer(1, 12))
    
    # Introduction
    intro_style = styles['Normal']
    intro_text = "This report provides a sentiment analysis of tweets from various X creators, identifying overall sentiment and financial tickers mentioned."
    story.append(Paragraph(intro_text, intro_style))
    story.append(Spacer(1, 24))
    
    # Report content
    for creator, details in data.items():
        # Creator Heading
        creator_heading = ParagraphStyle(
            'CreatorHeading',
            parent=styles['Heading2'],
            fontSize=16,
            leading=20
        )
        story.append(Paragraph(f"Analysis for {creator}", creator_heading))
        story.append(Spacer(1, 6))
        
        # Overall Sentiment
        sentiment_text = f"<b>Overall Sentiment:</b> {details['overall_sentiment']}"
        story.append(Paragraph(sentiment_text, intro_style))
        story.append(Spacer(1, 6))
        
        # Tickers
        tickers_text = "<b>Identified Tickers:</b>"
        story.append(Paragraph(tickers_text, intro_style))
        
        for ticker_info in details['tickers']:
            ticker_para = f"- {ticker_info['ticker']}: {ticker_info['direction']}"
            story.append(Paragraph(ticker_para, intro_style))
        
        story.append(Spacer(1, 18))
        
    doc.build(story)
    
    return f"PDF report successfully created at: {file_path}"


# ==============================================================================
# 3. DEFINE YOUR AGENTS
# ==============================================================================
# Agents are your team of experts. Give each a clear role, goal, and backstory.

data_collector = Agent(
  role='X Data Collector',
  goal='Scrape tweets from 10 specified X creators and provide the data in JSON format.',
  backstory='You are a master of web scraping, specializing in social media platforms. Your job is to accurately and efficiently collect public data.',
  verbose=True,
  llm=llm_model,
  tools=[x_data_scraper_tool]
)

sentiment_analyzer = Agent(
  role='Sentiment Analyzer',
  goal='Analyze collected tweets to determine sentiment and extract financial tickers and their direction (e.g., bullish or bearish).',
  backstory='You are a seasoned financial analyst with an eye for detail. You can read social media text and infer market sentiment and stock trends.',
  verbose=True,
  llm=llm_model,
)

output_arranger = Agent(
  role='Output Arranger',
  goal='Take the analyzed data and structure it into a final, clean JSON format for the report.',
  backstory='You are a meticulous data engineer who ensures all data is perfectly structured and ready for presentation. Your work is key to producing a clear and accurate report.',
  verbose=True,
  llm=llm_model,
)

report_generator = Agent(
  role='Report Generator',
  goal='Create a professional PDF report summarizing the findings for each creator.',
  backstory='You are a professional report writer and business analyst. Your final product is a clear, concise, and insightful document.',
  verbose=True,
  llm=llm_model,
  tools=[pdf_generator_tool]
)

# ==============================================================================
# 4. DEFINE YOUR TASKS
# ==============================================================================
# Tasks are the specific actions the agents perform. 'context' links the output
# of one task to the next.

scrape_task = Task(
  description='Scrape the latest 200 tweets for the following X creators: CreatorA, CreatorB, CreatorC, CreatorD, CreatorE. Use the scraping tool to perform this action. The list of users to scrape is provided as a comma-separated string.',
  agent=data_collector,
  expected_output='A JSON string containing the tweets from the specified creators.',
)

analysis_task = Task(
  description='Analyze the JSON data provided by the scraper. For each creator, determine their overall sentiment (e.g., "bullish", "bearish", or "neutral") and identify any financial tickers mentioned, along with the implied direction (e.g., $GOOG - bullish).',
  agent=sentiment_analyzer,
  context=[scrape_task],
  expected_output='A JSON string containing an object for each creator, including their overall sentiment and a list of identified tickers with their direction. Example: {"CreatorA": {"overall_sentiment": "bullish", "tickers": [{"ticker": "$GOOG", "direction": "bullish"}, {"ticker": "$NVDA", "direction": "neutral"}]}}',
)

arrange_task = Task(
  description='Take the output from the sentiment analysis and arrange it into a clean, final JSON structure. The output should be a single JSON object where each key is a creator handle and the value is an object containing their "overall_sentiment" and a "tickers" list.',
  agent=output_arranger,
  context=[analysis_task],
  expected_output='A perfectly formatted JSON object with the final, structured results.',
)

report_task = Task(
  description='Generate a professional PDF report from the final structured JSON data. Use the PDF generator tool to create a file named "Sentiment_Report.pdf". The report must be well-organized and easy to read.',
  agent=report_generator,
  context=[arrange_task],
  expected_output='A success message indicating the PDF file has been created and its location.',
)

# ==============================================================================
# 5. CREATE THE CREW AND RUN IT!
# ==============================================================================
# The crew orchestrates the entire workflow. 'Process.sequential' ensures tasks
# run in a specific order, which is perfect for this project.

financial_crew = Crew(
  agents=[data_collector, sentiment_analyzer, output_arranger, report_generator],
  tasks=[scrape_task, analysis_task, arrange_task, report_task],
  process=Process.sequential,
  verbose=True # Corrected from '2' to True
)

# Kickoff the crew to start the process.
result = financial_crew.kickoff()

print("\n\n################################################################################")
print("## Crew has finished its work! Here is the final result:")
print("################################################################################")
print(result)
