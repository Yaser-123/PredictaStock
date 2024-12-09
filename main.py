# Import necessary libraries
import streamlit as st
from datetime import date
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
from prophet import Prophet  # Correct import for Prophet

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

# Select stock from dropdown
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.text_input('Enter Stock Ticker (e.g., GOOG, AAPL, TSLA):', value='GOOG')

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Function to load stock data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Show raw stock data
st.subheader('Raw Data')
st.write(data.tail())

def Candlestick_Chart():
    try:
        # Dynamically find the required columns
        open_column = [col for col in data.columns if 'Open' in col][0]
        high_column = [col for col in data.columns if 'High' in col][0]
        low_column = [col for col in data.columns if 'Low' in col][0]
        close_column = [col for col in data.columns if 'Close' in col][0]
        date_column = [col for col in data.columns if 'Date' in col][0]

        # Drop rows with NaN in the required columns
        clean_data = data.dropna(subset=[open_column, high_column, low_column, close_column, date_column])

        if clean_data.empty:
            st.warning("No valid data available for the candlestick chart.")
            return

        # Plot the candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=clean_data[date_column],
            open=clean_data[open_column],
            high=clean_data[high_column],
            low=clean_data[low_column],
            close=clean_data[close_column],
            increasing_line_color='green',
            decreasing_line_color='red'
        )])

        # Update layout with X-axis slider (range slider feature)
        fig.update_layout(
            title=f"{selected_stock} Candlestick Chart",
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(label="YTD", step="year", stepmode="todate"),
                        dict(label="All", step="all")
                    ])
                ),
                rangeslider=dict(visible=True),  # Shows the X-axis slider
                type="date"
            ),
            template="plotly_dark"
        )

        st.plotly_chart(fig)

    except IndexError:
        st.error("One or more required columns (Open, High, Low, Close, Date) are missing in the data.")
    except Exception as e:
        st.error(f"An error occurred while creating the candlestick chart: {e}")


# Subheading for visualization
st.subheader('Candlestick Chart')
Candlestick_Chart()



# Plot time series data with matplotlib
def plot_time_series():
    # Dynamically find the columns containing 'Open' and 'Close'
    open_column = [col for col in data.columns if 'Open' in col][0]
    close_column = [col for col in data.columns if 'Close' in col][0]

    try:
        # Drop rows with NaN in Open or Close
        clean_data = data.dropna(subset=[open_column, close_column])
    except KeyError:
        st.error("Required columns not found in the DataFrame.")
        return

    # Plot the data using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=clean_data['Date'], y=clean_data[open_column],
                         mode='lines', name='Open Price', line_color='green'))
    fig.add_trace(go.Scatter(x=clean_data['Date'], y=clean_data[close_column],
                         mode='lines', name='Close Price', line_color='blue'))

    # Enable the X-axis slider (range slider feature)
    fig.update_layout(
        title=f"Time Series of {selected_stock} Stock (Open and Close Prices)",
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(label="YTD", step="year", stepmode="todate"),
                    dict(label="All", step="all")
                ])
            ),
            rangeslider=dict(visible=True),  # Shows the X slider
            type="date"
        ),
        template="plotly_dark"
    )

    st.plotly_chart(fig)

# Subheading for visualization
st.subheader('Time Series Chart')
plot_time_series()

# Clean up column names by stripping spaces
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join(col).strip() for col in data.columns]

data.columns = data.columns.str.strip()

# Prepare the data for Prophet
try:
    # Dynamically select the column based on the selected stock
    close_column = f'Close_{selected_stock}'  # For example, 'Close_GOOG' or 'Close_AAPL'
    date_column = 'Date_'  # Assuming 'Date_' column

    # Select relevant columns to train the model
    df_train = data[[date_column, close_column]].copy()

    # Rename columns as required by Prophet
    df_train = df_train.rename(columns={date_column: "ds", close_column: "y"})

    # Convert 'Close' column to numeric, coercing errors to NaN
    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')

    # Drop rows where 'Close' is NaN after conversion
    df_train.dropna(subset=['y'], inplace=True)

    # Initialize and fit the Prophet model
    m = Prophet()
    m.fit(df_train)

    # Create a future DataFrame for the forecast
    period = 365  # Predict for 1 year
    future = m.make_future_dataframe(periods=period)

    # Predict future prices
    forecast = m.predict(future)

    # Show all components in the forecast DataFrame
    st.subheader('Forecast Data')
    st.write(forecast)  # Display everything Prophet outputs (Date, yhat, uncertainty bounds, etc.)

except KeyError as e:
    st.error(f"KeyError: {e}. Check if the required columns exist in your data.")

# Plot forecast dynamically
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

# Add Social Sharing Buttons
st.subheader("Share Your Analysis")
social_buttons_html = """
<div style="display: flex; gap: 10px;">
    <a href="https://www.facebook.com/sharer/sharer.php?u=https://predictastock.streamlit.app/&quote=Check%20out%20this%20amazing%20Stock%20Prediction%20App!%20📈🔥" target="_blank">
        <button style="background-color:#3b5998;color:white;border:none;padding:10px 20px;border-radius:5px;">Share on Facebook</button>
    </a>
    <a href="https://www.linkedin.com/shareArticle?url=https://predictastock.streamlit.app/&title=Amazing%20Stock%20Prediction%20App!&summary=Get%20accurate%20stock%20predictions%20and%20analyze%20trends%20with%20our%20tool.%20Try%20it%20now!" target="_blank">
        <button style="background-color:#0077b5;color:white;border:none;padding:10px 20px;border-radius:5px;">Share on LinkedIn</button>
    </a>
    <a href="https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20Stock%20Prediction%20App!%20📈🔥&url=https://predictastock.streamlit.app/" target="_blank">
        <button style="background-color:#1DA1F2;color:white;border:none;padding:10px 20px;border-radius:5px;">Share on Twitter</button>
    </a>
    <a href="https://www.instagram.com/" target="_blank">
        <button style="background-color:#bc2a8d;color:white;border:none;padding:10px 20px;border-radius:5px;">Share on Instagram</button>
    </a>
</div>
"""
st.markdown(social_buttons_html, unsafe_allow_html=True)

