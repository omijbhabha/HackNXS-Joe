import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

def scrape_news(stock_symbol):
    base_url = f"https://finance.yahoo.com/quote/{stock_symbol}/news"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = []
    for headline in soup.find_all('h3'):
        headlines.append(headline.text)
    return headlines

def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

def main():
    stock_symbol = input("Enter the stock symbol you want to analyze (e.g., AAPL): ")
    news_headlines = scrape_news(stock_symbol)
    sentiments = [get_sentiment(headline) for headline in news_headlines]
    
    positive_count = sentiments.count('positive')
    negative_count = sentiments.count('negative')
    neutral_count = sentiments.count('neutral')
    total_headlines = len(news_headlines)
    print(f"Out of {total_headlines} news headlines:")
    print(f"Positive sentiment: {positive_count} headlines")
    print(f"Negative sentiment: {negative_count} headlines")
    print(f"Neutral sentiment: {neutral_count} headlines")

if __name__ == "__main__":
    main()
