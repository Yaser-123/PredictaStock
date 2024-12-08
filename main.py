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
selected_stock = st.selectbox('Select dataset for prediction', stocks)

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
