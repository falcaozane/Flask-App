from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import concurrent.futures

app = Flask(__name__)

# Function to fetch historical stock data
def get_data(assets, start_date, end_date):
    df = pd.DataFrame()
    for stock in assets:
        df[stock] = yf.download(stock, start=start_date, end=end_date)['Adj Close']
    return df

# Function to perform Monte Carlo simulation
def monte_carlo(df, assets, num_of_portfolios):
    log_returns = np.log(1 + df.pct_change())
    all_weights = np.zeros((num_of_portfolios, len(assets)))
    ret_arr = np.zeros(num_of_portfolios)
    vol_arr = np.zeros(num_of_portfolios)
    sharpe_arr = np.zeros(num_of_portfolios)
    
    for i in range(num_of_portfolios):
        weights = np.random.random(len(assets))
        weights /= np.sum(weights)
        all_weights[i, :] = weights
        ret_arr[i] = np.sum(log_returns.mean() * weights) * 252
        vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
        sharpe_arr[i] = ret_arr[i] / vol_arr[i]
    
    return pd.DataFrame({'Returns': ret_arr, 'Volatility': vol_arr, 'Sharpe Ratio': sharpe_arr, 'Weights': list(all_weights)})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize_portfolio', methods=['POST'])
def optimize_portfolio():
    data = request.get_json()
    
    # Extracting data from the request
    age = data['age']
    risk_tolerance = int(data['risk_tolerance'])
    annual_salary = float(data['salary'])
    savings = float(data['savings'])
    desired_amount = float(data['desired_amount'])
    assets = data['assets']

    # Fixed start and end dates
    start_date = datetime(2014, 1, 1)
    end_date = datetime.today()

    # Fetch data
    df = get_data(assets, start_date, end_date)

    # Portfolio statistics
    weights = np.array([1/len(assets)] * len(assets))
    returns = df.pct_change()
    annual_return = np.sum(returns.mean() * weights) * 252
    cov_matrix_annual = returns.cov() * 252
    port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
    port_volatility = np.sqrt(port_variance)

    # Monte Carlo Simulation
    num_of_portfolios = 5000
    sim_df = monte_carlo(df, assets, num_of_portfolios)
    max_sharpe_ratio = sim_df.loc[sim_df['Returns'].idxmax()]
    min_volatility = sim_df.loc[sim_df['Volatility'].idxmin()]

    # Optimized Portfolio
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)
    ef = EfficientFrontier(mu, S)

    # Calculate minimum volatility portfolio
    ef_min_volatility = EfficientFrontier(mu, S)
    min_vol_weights = ef_min_volatility.min_volatility()
    min_vol_performance = ef_min_volatility.portfolio_performance()
    min_volatility = min_vol_performance[1]

    # Calculate maximum Sharpe ratio portfolio
    ef_max_sharpe = EfficientFrontier(mu, S)
    max_sharpe_weights = ef_max_sharpe.max_sharpe()
    max_sharpe_performance = ef_max_sharpe.portfolio_performance()
    max_sharpe_volatility = max_sharpe_performance[1]

    # Adjust portfolio optimization based on risk tolerance and investment period
    investment_period = 60 - int(age)
    try:
        if risk_tolerance <= 3:
            # Low risk tolerance: minimize volatility
            raw_weights = ef.min_volatility()
        elif risk_tolerance >= 8:
            # High risk tolerance: maximize Sharpe ratio
            raw_weights = ef.max_sharpe()
        else:
            # Intermediate risk tolerance: target volatility
            # Linear interpolation between minimum and maximum volatility
            target_volatility = min_volatility + (max_sharpe_volatility - min_volatility) * ((risk_tolerance - 3) / 4)

            # Adjust target volatility based on investment period
            adjustment_factor = max(0.5, investment_period / 30)  # Scale down volatility for shorter investment periods
            target_volatility *= adjustment_factor

            raw_weights = ef.efficient_risk(target_volatility)
    except ValueError as e:
        # In case of error, fallback to minimum volatility portfolio
        raw_weights = min_vol_weights

    cleaned_weights = ef.clean_weights()

    # Ensure the weights sum to 1 (or 100%)
    weight_sum = sum(cleaned_weights.values())
    if weight_sum != 1:
        cleaned_weights = {k: v / weight_sum for k, v in cleaned_weights.items()}

    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
    optimized_weights_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight'])
    print(optimized_weights_df)
    optimized_weights_dict = optimized_weights_df['Weight'].to_dict()

    return jsonify({
        'expected_annual_return': expected_annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'optimized_weights': optimized_weights_dict
    })

if __name__ == '__main__':
    app.run(debug=True,port=8080)
