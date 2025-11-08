"""
Market Sectors Real-Time Prediction Dashboard
Shows sentiment analysis and predictions for multiple market sectors
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

from src.pipeline.sector_predictor import SectorPredictor
from src.pipeline.sector_config import SECTORS


# Page config
st.set_page_config(
    page_title="SOOIQ - Market Sectors Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1E88E5;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .sector-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    
    .tech-card {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
    }
    
    .finance-card {
        background: linear-gradient(135deg, #27AE60 0%, #1E8449 100%);
    }
    
    .energy-card {
        background: linear-gradient(135deg, #E67E22 0%, #D35400 100%);
    }
    
    .sentiment-box {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .signal-buy {
        background: #27AE60;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
    }
    
    .signal-sell {
        background: #E74C3C;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
    }
    
    .signal-hold {
        background: #F39C12;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load sector predictor (cached)"""
    return SectorPredictor()


def plot_sentiment_gauge(sentiment_score: float, title: str = "Sector Sentiment") -> go.Figure:
    """Create a gauge chart for sentiment"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sentiment_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#1E88E5"},
            'steps': [
                {'range': [0, 33], 'color': "#ffebee"},
                {'range': [33, 66], 'color': "#fff9c4"},
                {'range': [66, 100], 'color': "#e8f5e9"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig


def display_sector_card(sector_result: dict):
    """Display a sector analysis card"""
    sector_info = sector_result['sector_info']
    sentiment = sector_result['sector_sentiment']
    predictions = sector_result['ticker_predictions']
    
    # Determine card class
    card_class = f"{sector_info.name}-card"
    
    st.markdown(f"""
    <div class="sector-card {card_class}">
        <h2>{sector_info.icon} {sector_info.display_name} Sector</h2>
        <div class="sentiment-box">
            <h3>Sentiment Score: {sentiment['sentiment_score']:.1%}</h3>
            <p>Based on {sentiment['total_articles']} news articles</p>
            <p>
                <span style="color: #00cc00;">✓ Positive: {sentiment['positive_count']}</span> | 
                <span style="color: #ff4444;">✗ Negative: {sentiment['negative_count']}</span> | 
                <span style="color: #ffaa00;">○ Neutral: {sentiment['neutral_count']}</span>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sentiment gauge
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig = plot_sentiment_gauge(sentiment['sentiment_score'], "Sentiment")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top news headlines
        st.markdown("### 📰 Top News Headlines")
        if sentiment['articles']:
            for i, article in enumerate(sentiment['articles'][:3], 1):
                sentiment_color = {
                    'positive': '#00cc00',
                    'negative': '#ff4444',
                    'neutral': '#ffaa00'
                }
                color = sentiment_color.get(article['label'], '#666')
                
                st.markdown(f"""
                <div style="background: white; padding: 0.8rem; border-radius: 5px; 
                            margin: 0.5rem 0; border-left: 4px solid {color};">
                    <b>{i}. {article['title']}</b><br>
                    <small>Source: {article.get('source', 'Unknown')} | 
                    Sentiment: <span style="color: {color};">{article['label'].upper()}</span> 
                    ({article['confidence']:.1%})</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent news available")
    
    # Ticker predictions
    st.markdown("### 📊 Stock Predictions")
    
    cols = st.columns(3)
    
    for idx, pred in enumerate(predictions):
        with cols[idx]:
            signal_class = f"signal-{pred['signal'].lower()}"
            
            # Determine arrow based on price change
            price_change = pred.get('price_change_7d', 0)
            arrow = "↑" if price_change > 0 else "↓" if price_change < 0 else "→"
            price_color = "#27AE60" if price_change > 0 else "#E74C3C" if price_change < 0 else "#666"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{pred['ticker']}</h3>
                <p style="font-size: 0.9rem; color: #666;">{pred.get('display_name', pred['ticker'])}</p>
                <div class="{signal_class}" style="margin: 0.5rem 0;">
                    {pred['signal']}
                </div>
                <p><b>Confidence:</b> {pred['confidence']:.1%}</p>
                <p><b>Price:</b> ${pred.get('current_price', 0):.2f} 
                   <span style="color: {price_color};">{arrow} {abs(price_change):.2%}</span></p>
                <p><b>Risk:</b> {pred.get('risk_level', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main dashboard"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 SOOIQ Market Sectors Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.2rem; color: #666;">
    Real-Time Sector Analysis & Stock Predictions Powered by AI
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Dashboard Controls")
    
    # Sector selection
    st.sidebar.subheader("📈 Select Sectors to Analyze")
    
    sector_options = {
        info.display_name: name 
        for name, info in SECTORS.items()
    }
    
    selected_sectors = st.sidebar.multiselect(
        "Choose sectors:",
        options=list(sector_options.keys()),
        default=list(sector_options.keys())  # All selected by default
    )
    
    # Refresh controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Update Controls")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        refresh_button = st.button("🔄 Refresh All", use_container_width=True)
    with col2:
        clear_cache = st.button("🗑️ Clear Cache", use_container_width=True)
    
    use_cache = st.sidebar.checkbox("Use cached data (faster)", value=True)
    
    # Info
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **How it works:**
    1. Fetches real-time news for each sector
    2. Analyzes sentiment using FinBERT AI
    3. Generates predictions for top stocks
    4. Updates every 5 minutes (cached)
    """)
    
    # Load predictor
    predictor = load_predictor()
    
    # Clear cache if requested
    if clear_cache:
        predictor.clear_cache()
        st.sidebar.success("✓ Cache cleared!")
        st.rerun()
    
    # Main content
    if not selected_sectors:
        st.warning("⚠️ Please select at least one sector from the sidebar")
        return
    
    # Show loading message
    if refresh_button or not use_cache:
        st.info("🔄 Fetching latest news and analyzing sectors... This may take a minute.")
    
    # Analyze sectors
    with st.spinner("Analyzing market sectors..."):
        for sector_display_name in selected_sectors:
            sector_name = sector_options[sector_display_name]
            
            try:
                result = predictor.predict_sector(sector_name, use_cache=use_cache and not refresh_button)
                
                # Display sector card
                display_sector_card(result)
                
                # Add separator
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error analyzing {sector_display_name}: {e}")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sectors Analyzed", len(selected_sectors))
    with col2:
        st.metric("Total Stocks", len(selected_sectors) * 3)
    with col3:
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("""
    <p style="text-align: center; color: #999; margin-top: 2rem;">
    SOOIQ Stock Predictor - Hackathon Demo | Powered by FinBERT & NewsAPI
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
