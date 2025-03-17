"""import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Configurations ---
ALPHA_VANTAGE_API_KEY = "Your_API_Key"

# --- Fetch AI Scores (Alternative to Danelfin) ---
def get_alpha_vantage_scores(ticker):
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url).json()
    try:
        return float(response.get("AnalystTargetPrice", 50))  # Default 50 if unavailable
    except (TypeError, ValueError):
        return 50  # Fallback default

# --- Get Market Sentiment using News Sentiment Analysis ---
def get_news_sentiment(ticker):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url).json()
    
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    
    for article in response.get("feed", []):
        sentiment = analyzer.polarity_scores(article.get("title", "") + " " + article.get("summary", ""))
        sentiment_scores.append(sentiment["compound"])
    
    return np.mean(sentiment_scores) if sentiment_scores else 0  # Default neutral if no news found

# --- Fetch Fundamental Data ---
def get_fundamental_data(ticker):
    url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url).json()
    
    try:
        financials = response.get("annualReports", [{}])[0]
        return {
            "Revenue": float(financials.get("totalRevenue", 0)),
            "Net Income": float(financials.get("netIncome", 0)),
            "Debt to Equity": float(financials.get("totalLiabilities", 1)) / max(1, float(financials.get("totalShareholderEquity", 1))),
            "Operating Cash Flow": float(financials.get("operatingCashflow", 0))
        }
    except (TypeError, ValueError):
        return {"Revenue": 0, "Net Income": 0, "Debt to Equity": 0, "Operating Cash Flow": 0}

# --- Fetch Technical Indicators for Trading Strategy ---
def get_technical_analysis(ticker):
    url = f"https://www.alphavantage.co/query?function=SMA&symbol={ticker}&interval=daily&time_period=50&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url).json()
    sma_values = response.get("Technical Analysis: SMA", {})
    try:
        latest_sma = float(list(sma_values.values())[0]["SMA"])
        return "BUY" if latest_sma > 0 else "SELL"
    except (IndexError, KeyError, ValueError):
        return "HOLD"

# --- Stock Selection and Prediction Model ---
def analyze_stock(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    if hist.empty:
        print(f"No historical data available for {ticker}")
        return None
    
    ai_score = get_alpha_vantage_scores(ticker)
    sentiment = get_news_sentiment(ticker)
    strategy = get_technical_analysis(ticker)
    financials = get_fundamental_data(ticker)
    
    scaler = MinMaxScaler()
    stock_data = scaler.fit_transform(hist[['Close']].values)
    prediction = np.mean(stock_data[-5:]) * 1.02  # Basic model: 2% growth estimate
    
    # Decision Logic for Buy/Sell/Hold
    if ai_score > 70 and sentiment > 0.2 and strategy == "BUY" and financials["Net Income"] > 0:
        decision = "BUY"
    elif ai_score < 40 or sentiment < -0.2 or strategy == "SELL" or financials["Net Income"] < 0:
        decision = "SELL"
    else:
        decision = "HOLD"
    
    result = {
        "Ticker": ticker,
        "AI Score": ai_score,
        "Sentiment Score": sentiment,
        "Trading Strategy": strategy,
        "Predicted Price": prediction,
        "Revenue": financials["Revenue"],
        "Net Income": financials["Net Income"],
        "Debt to Equity": financials["Debt to Equity"],
        "Operating Cash Flow": financials["Operating Cash Flow"],
        "Decision": decision
    }
    
    print("\nStock Analysis for", ticker)
    for key, value in result.items():
        print(f"{key}: {value}")
    
    return result

# --- Portfolio Analysis ---
stock_list = ["MSFT", "NVDA"]  # Example stocks
results = [analyze_stock(stock) for stock in stock_list if stock]

# Convert to DataFrame for better readability
portfolio_df = pd.DataFrame([r for r in results if r])
print("\n--- Portfolio Summary ---")
print(portfolio_df)"""


import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Alpha Vantage API Key (Replace 'YOUR_API_KEY' with actual key)
ALPHA_VANTAGE_API_KEY = "JO13YEMVD4ZD415D"
BASE_URL = "https://www.alphavantage.co/query"

# --- Get AI Score (Placeholder for Danelfin Alternative) ---
def get_ai_score(ticker):
    return np.random.uniform(30, 90)  # Simulated AI score between 30-90

# --- Get Market Sentiment using News Sentiment Analysis (Placeholder) ---
def get_news_sentiment(ticker):
    return np.random.uniform(-0.5, 0.5)  # Simulated sentiment score

# --- Fetch Fundamental Data from Alpha Vantage ---
def get_fundamental_data(ticker):
    url = f"{BASE_URL}?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url).json()
    
    try:
        return {
            "Revenue": float(response.get("RevenueTTM", 0)),
            "Net Income": float(response.get("NetIncomeTTM", 0)),
            "Debt to Equity": float(response.get("DebtToEquity", 1)),
            "Operating Cash Flow": float(response.get("OperatingCashflowTTM", 0))
        }
    except (TypeError, ValueError):
        return {"Revenue": 0, "Net Income": 0, "Debt to Equity": 0, "Operating Cash Flow": 0}

# --- Fetch Technical Indicators for Trading Strategy ---
def get_technical_analysis(ticker):
    url = f"{BASE_URL}?function=SMA&symbol={ticker}&interval=daily&time_period=50&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url).json()
    
    try:
        sma_50 = float(response["Technical Analysis: SMA"][list(response["Technical Analysis: SMA"].keys())[0]]["SMA"])
        
        url_price = f"{BASE_URL}?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        latest_price = float(requests.get(url_price).json()["Global Quote"]["05. price"])
        
        if latest_price > sma_50:
            return "BUY"
        elif latest_price < sma_50:
            return "SELL"
        else:
            return "HOLD"
    except (KeyError, ValueError, TypeError):
        return "HOLD"

# --- Stock Selection and Prediction Model ---
def analyze_stock(ticker):
    url = f"{BASE_URL}?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url).json()
    
    if "Time Series (Daily)" not in response:
        print(f"\n⚠️ API Error for {ticker}: {response}")
        print("Possible reasons: API limits, incorrect API key, or missing data.")
        return None
    
    try:
        hist_data = response["Time Series (Daily)"]
        hist_prices = [float(hist_data[day]["4. close"]) for day in list(hist_data.keys())[:252]]
        
        ai_score = get_ai_score(ticker)
        sentiment = get_news_sentiment(ticker)
        strategy = get_technical_analysis(ticker)
        financials = get_fundamental_data(ticker)
        
        scaler = MinMaxScaler()
        stock_data = scaler.fit_transform(np.array(hist_prices).reshape(-1, 1))
        prediction = np.mean(stock_data[-5:]) * 1.02  # Basic model: 2% growth estimate
        
        # Decision Logic for Buy/Sell/Hold
        if ai_score > 70 and sentiment > 0.2 and strategy == "BUY" and financials["Net Income"] > 0:
            decision = "BUY"
        elif ai_score < 40 or sentiment < -0.2 or strategy == "SELL" or financials["Net Income"] < 0:
            decision = "SELL"
        else:
            decision = "HOLD"
        
        result = {
            "Ticker": ticker,
            "AI Score": ai_score,
            "Sentiment Score": sentiment,
            "Trading Strategy": strategy,
            "Predicted Price": prediction,
            "Revenue": financials["Revenue"],
            "Net Income": financials["Net Income"],
            "Debt to Equity": financials["Debt to Equity"],
            "Operating Cash Flow": financials["Operating Cash Flow"],
            "Decision": decision
        }
        
        print("\nStock Analysis for", ticker)
        for key, value in result.items():
            print(f"{key}: {value}")
        
        return result
    except (KeyError, ValueError, TypeError):
        print(f"No historical data available for {ticker}")
        return None

# --- Portfolio Analysis ---
stock_list = ["AAPL", "MSFT", "TSLA", "NVDA"]  # Example stocks
results = [analyze_stock(stock) for stock in stock_list if stock]

# Convert to DataFrame for better readability
portfolio_df = pd.DataFrame([r for r in results if r])
print("\n--- Portfolio Summary ---")
print(portfolio_df)
