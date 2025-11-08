"""
SOOIQ Stock Predictor - Streamlit Demo App
24-Hour Hackathon Version
"""

import os
import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

# Suppress Pydantic settings errors
os.environ.setdefault('SUPPORTED_MARKETS', 'US')

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.demo.simple_predictor import SimpleStockPredictor

# Page config
st.set_page_config(
    page_title="SOOIQ Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .buy-signal {
        color: #00cc00;
        font-weight: bold;
        font-size: 2rem;
    }
    .sell-signal {
        color: #ff4444;
        font-weight: bold;
        font-size: 2rem;
    }
    .hold-signal {
        color: #ffaa00;
        font-weight: bold;
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize predictor (cached)
@st.cache_resource
def load_predictor():
    """Load the predictor model (cached to avoid reloading)"""
    return SimpleStockPredictor()


def plot_price_history(price_data):
    """Plot price history with Plotly"""
    hist = price_data['history']
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name='Price'
    ))
    
    fig.update_layout(
        title='30-Day Price History',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        height=400,
        template='plotly_white',
        xaxis_rangeslider_visible=False
    )
    
    return fig


def plot_sentiment_breakdown(sentiment_breakdown):
    """Plot sentiment breakdown pie chart"""
    labels = ['Positive', 'Negative', 'Neutral']
    values = [
        sentiment_breakdown['positive'],
        sentiment_breakdown['negative'],
        sentiment_breakdown['neutral']
    ]
    colors = ['#00cc00', '#ff4444', '#ffaa00']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.3
    )])
    
    fig.update_layout(
        title='News Sentiment Breakdown',
        height=300
    )
    
    return fig


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 SOOIQ Stock Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.2rem; color: #666;">
    AI-Powered Stock Analysis using Sentiment & Price Momentum
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Popular stocks for quick selection
    st.sidebar.subheader("🔥 Popular Stocks")
    popular_stocks = {
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Google': 'GOOGL',
        'Tesla': 'TSLA',
        'NVIDIA': 'NVDA',
        'Amazon': 'AMZN',
        'Meta': 'META',
        'Netflix': 'NFLX'
    }
    
    selected_popular = st.sidebar.selectbox(
        "Select a popular stock:",
        options=['Custom...'] + list(popular_stocks.keys())
    )
    
    # Stock input
    if selected_popular == 'Custom...':
        ticker = st.sidebar.text_input(
            "Enter Stock Ticker:",
            value="AAPL",
            help="Enter any valid stock ticker (e.g., AAPL, MSFT, TSLA)"
        ).upper()
    else:
        ticker = popular_stocks[selected_popular]
        st.sidebar.info(f"Selected: {ticker}")
    
    # Analyze button
    analyze_button = st.sidebar.button("🔍 Analyze Stock", type="primary", use_container_width=True)
    
    # Info section
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info("""
    **SOOIQ** combines:
    - 📰 News Sentiment (FinBERT AI)
    - 📊 Price Momentum
    - 🤖 Smart Prediction Logic
    
    **Hackathon Version**
    Built in 24 hours! 🚀
    """)
    
    # Main content
    if analyze_button:
        if not ticker:
            st.error("Please enter a stock ticker!")
            return
        
        try:
            # Load predictor
            with st.spinner("Loading AI model..."):
                predictor = load_predictor()
            
            # Make prediction
            with st.spinner(f"Analyzing {ticker}... This may take a moment..."):
                result = predictor.predict(ticker)
            
            # Check for errors
            if 'error' in result:
                st.error(f"❌ Error: {result['error']}")
                st.info("💡 Tip: Make sure the ticker is valid and try again.")
                return
            
            # Display results
            st.success(f"✅ Analysis complete for **{ticker}**")
            
            # Signal banner
            signal = result['signal']
            signal_class = f"{signal.lower()}-signal"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 1rem; margin: 1rem 0;">
                <h2 style="color: white; margin: 0;">Recommendation</h2>
                <div class="{signal_class}" style="color: white; margin-top: 1rem;">{signal}</div>
                <p style="color: white; margin-top: 0.5rem;">Confidence: {result['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${result['current_price']:.2f}"
                )
            
            with col2:
                st.metric(
                    "7-Day Change",
                    f"{result['price_change_7d']:+.2%}",
                    delta=f"{result['price_change_7d']:+.2%}"
                )
            
            with col3:
                st.metric(
                    "Sentiment Score",
                    f"{result['sentiment_score']:.1%}",
                    delta="Positive" if result['sentiment_score'] > 0.5 else "Negative"
                )
            
            with col4:
                risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
                st.metric(
                    "Risk Level",
                    f"{risk_color.get(result['risk_level'], '⚪')} {result['risk_level']}"
                )
            
            # Charts row
            st.markdown("---")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Price history chart
                price_data = predictor.get_price_data(ticker)
                if price_data:
                    fig_price = plot_price_history(price_data)
                    st.plotly_chart(fig_price, use_container_width=True)
            
            with col_chart2:
                # Sentiment breakdown
                fig_sentiment = plot_sentiment_breakdown(result['sentiment_breakdown'])
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            # Detailed metrics
            st.markdown("---")
            st.subheader("📊 Detailed Analysis")
            
            col_detail1, col_detail2 = st.columns(2)
            
            with col_detail1:
                st.markdown("**Price Metrics**")
                st.write(f"- 30-Day Change: {result['price_change_30d']:+.2%}")
                st.write(f"- Momentum Score: {result['momentum_score']:+.3f}")
                st.write(f"- Volatility: {result['volatility']:.4f}")
                st.write(f"- Volume Ratio: {result['volume_ratio']:.2f}x")
            
            with col_detail2:
                st.markdown("**Sentiment Metrics**")
                st.write(f"- Positive News: {result['sentiment_breakdown']['positive']}")
                st.write(f"- Negative News: {result['sentiment_breakdown']['negative']}")
                st.write(f"- Neutral News: {result['sentiment_breakdown']['neutral']}")
                st.write(f"- Total Articles: {result['news_count']}")
            
            # Sample news
            if result['sample_news']:
                st.markdown("---")
                st.subheader("📰 Real-Time News Analysis")
                st.markdown(f"*Analyzing {result['news_count']} recent articles from NewsAPI*")
                
                for i, article in enumerate(result['sample_news'], 1):
                    sentiment_emoji = {
                        'positive': '�',
                        'negative': '�',
                        'neutral': '�'
                    }
                    emoji = sentiment_emoji.get(article['label'], '📄')
                    
                    # Color based on sentiment
                    sentiment_color = {
                        'positive': '#00cc00',
                        'negative': '#ff4444',
                        'neutral': '#ffaa00'
                    }
                    color = sentiment_color.get(article['label'], '#666')
                    
                    with st.expander(f"{emoji} **{article.get('title', article['text'][:80])}**"):
                        # Sentiment badge
                        st.markdown(f"""
                        <div style="background: {color}; color: white; padding: 5px 10px; 
                                    border-radius: 5px; display: inline-block; margin-bottom: 10px;">
                            {article['label'].upper()} - {article['confidence']:.1%} confidence
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Article metadata
                        if article.get('source'):
                            st.write(f"**Source:** {article.get('source', 'Unknown')}")
                        if article.get('published_at'):
                            st.write(f"**Published:** {article['published_at'][:10]}")
                        
                        # Article text/description
                        st.write(article['text'])
                        
                        # Link to full article
                        if article.get('url'):
                            st.markdown(f"[Read full article →]({article['url']})")
            
            
            # Timestamp
            st.markdown("---")
            st.caption(f"Analysis generated at: {result['timestamp']}")
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("💡 Try a different ticker or check your internet connection.")
            import traceback
            with st.expander("🔍 Error Details (for debugging)"):
                st.code(traceback.format_exc())
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to SOOIQ Stock Predictor!
        
        ### How it works:
        
        1. **📰 News Analysis**: We analyze recent news articles using FinBERT, a state-of-the-art AI model trained on financial text
        2. **📊 Price Momentum**: We calculate price trends and momentum indicators from historical data
        3. **🤖 Smart Prediction**: Our algorithm combines sentiment and momentum to generate Buy/Hold/Sell signals
        
        ### Get Started:
        
        👈 Select a stock from the sidebar and click **Analyze Stock** to begin!
        
        ---
        
        ### ⚡ Quick Demo Stocks:
        """)
        
        # Quick demo buttons
        col_demo1, col_demo2, col_demo3, col_demo4 = st.columns(4)
        
        demo_stocks = [
            ('AAPL', 'Apple'),
            ('MSFT', 'Microsoft'),
            ('GOOGL', 'Google'),
            ('TSLA', 'Tesla')
        ]
        
        for col, (ticker_demo, name) in zip([col_demo1, col_demo2, col_demo3, col_demo4], demo_stocks):
            with col:
                st.info(f"**{name}**\n\n`{ticker_demo}`")
        
        st.markdown("---")
        
        # Feature highlights
        st.markdown("""
        ### 🌟 Features:
        
        - ✅ Real-time stock data from Yahoo Finance
        - ✅ AI-powered sentiment analysis (FinBERT)
        - ✅ Interactive price charts
        - ✅ Risk assessment
        - ✅ Confidence scores
        - ✅ Multiple popular stocks supported
        
        ### 🚀 Built With:
        
        - **Python** + **PyTorch**
        - **FinBERT** (HuggingFace)
        - **Streamlit** (UI)
        - **Plotly** (Charts)
        - **yfinance** (Data)
        
        ---
        
        <p style="text-align: center; color: #666;">
        🎯 Built for 24-hour hackathon sprint | 💡 Powered by AI
        </p>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
