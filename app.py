import streamlit as st
import subprocess
import sys
import os
from datetime import datetime

def main():
    st.set_page_config(
        page_title="CrewAI Sentiment Analysis",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 CrewAI Sentiment Analysis Demo")
    st.markdown("---")
    
    st.markdown("""
    ### About This Project
    This is a **CrewAI-powered sentiment analysis system** that analyzes X (Twitter) creators' content.
    
    **Features:**
    - 🕵️ **X Data Collector Agent**: Simulates data collection from X creators
    - 📊 **Sentiment Analyzer Agent**: Analyzes sentiment using VADER and TextBlob
    - 📈 **Financial Ticker Extractor**: Identifies stock/crypto mentions
    - 📋 **Report Generator Agent**: Creates comprehensive reports
    
    **No API Keys Required!** This demo uses simulated data to showcase the full workflow.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### How It Works")
        st.markdown("""
        1. **Data Collection**: Simulates scraping 10 X creators with 200+ tweets each
        2. **Sentiment Analysis**: Processes each tweet for sentiment and topics
        3. **Financial Analysis**: Extracts stock tickers and crypto mentions
        4. **Report Generation**: Creates detailed PDF and text reports
        """)
    
    with col2:
        st.markdown("### Demo Creators")
        st.markdown("""
        - @elonmusk
        - @sundarpichai
        - @tim_cook
        - @satyanadella
        - @paraga
        - @jack
        - @brianchesky
        - @dhh
        - @naval
        - @pmarca
        """)
    
    st.markdown("---")
    
    # Run Analysis Button
    if st.button("🚀 Run CrewAI Analysis", type="primary", use_container_width=True):
        with st.spinner("🤖 CrewAI agents are working..."):
            try:
                # Run the working demo
                result = subprocess.run([sys.executable, "working_demo.py"], 
                                      capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    st.success("✅ Analysis Complete!")
                    
                    # Display the output
                    st.markdown("### 📊 Analysis Results")
                    st.code(result.stdout, language="text")
                    
                    # Check for generated files
                    files_created = []
                    if os.path.exists("Demo_Report.txt"):
                        files_created.append("📄 Demo_Report.txt")
                    if os.path.exists("Sentiment_Analysis_Report.pdf"):
                        files_created.append("📄 Sentiment_Analysis_Report.pdf")
                    
                    if files_created:
                        st.markdown("### 📁 Generated Files")
                        for file in files_created:
                            st.success(file)
                    
                else:
                    st.error("❌ Analysis failed!")
                    st.code(result.stderr, language="text")
                    
            except subprocess.TimeoutExpired:
                st.error("⏰ Analysis timed out. Please try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    # Project Information
    st.markdown("### 📚 Project Information")
    st.markdown("""
    **Technologies Used:**
    - 🐍 Python 3.8+
    - 🤖 CrewAI Framework
    - 📊 VADER Sentiment Analysis
    - 📈 TextBlob NLP
    - 📄 ReportLab PDF Generation
    
    **GitHub Repository:** [CrewAI Sentiment Analysis](https://github.com/Tanish-og/CREW_AI)
    
    **Created for:** Internship Qualification Assignment
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with ❤️ using CrewAI and Streamlit*")

if __name__ == "__main__":
    main()
