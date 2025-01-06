import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to load and preprocess data
def load_and_preprocess_data(file):
    data = pd.read_csv(file)
    data.rename(columns={"Sales (₹)": "Sales"}, inplace=True)
    data.fillna(0, inplace=True)

    # Define month order
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    # Convert Month to categorical type with the defined order
    data["Month"] = pd.Categorical(data["Month"], categories=month_order, ordered=True)

    return data

# Function to create graphs
def create_charts(data, selected_state=None, selected_product_category=None, selected_months=None):
    st.subheader("Sales Overview")

    # Filter data based on selected state
    if selected_state:
        data = data[data["State"] == selected_state]

    # Create subplots
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=("Total Sales by State", "Sales Distribution by Product Category",
                                        "Sales Trend by Month", "Average Sales by Product Category",
                                        "Total Sales by Month", "Sales by State and Product Category"),
                        specs=[[{"type": "bar"}, {"type": "pie"}, {"type": "scatter"}],
                               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]])

    # Total Sales by State
    sales_overview = data.groupby("State")["Sales"].sum().reset_index()
    sales_overview['Selected'] = sales_overview['State'].isin([selected_state])
    colors_state = np.where(sales_overview['Selected'], 'orange', 'royalblue')
    fig.add_trace(go.Bar(x=sales_overview['State'],
                         y=sales_overview['Sales'],
                         marker_color=colors_state),
                  row=1, col=1)

    # Sales Distribution by Product Category
    product_category_sales = data.groupby("Product_Category")["Sales"].sum().reset_index()
    product_category_sales['Selected'] = product_category_sales['Product_Category'].isin([selected_product_category])
    colors_category = np.where(product_category_sales['Selected'], 'orange', 'lightgray')
    fig.add_trace(go.Pie(labels=product_category_sales['Product_Category'],
                         values=product_category_sales['Sales'],
                         hole=0.3),
                  row=1, col=2)

    # Sales Trend by Month
    month_sales = data.groupby("Month")["Sales"].sum().reset_index()
    colors_month = ['orange' if month in selected_months else 'lightgray' for month in month_sales['Month']]
    fig.add_trace(go.Scatter(x=month_sales['Month'],
                             y=month_sales['Sales'],
                             mode='lines+markers',
                             marker=dict(color='orange')),
                  row=1, col=3)

    # Average Sales by Product Category
    avg_sales_per_category = data.groupby("Product_Category")["Sales"].mean().reset_index()
    avg_sales_per_category['Selected'] = avg_sales_per_category['Product_Category'].isin([selected_product_category])
    colors_avg_category = np.where(avg_sales_per_category['Selected'], 'orange', 'lightgreen')
    fig.add_trace(go.Bar(x=avg_sales_per_category['Product_Category'],
                         y=avg_sales_per_category['Sales'],
                         marker_color=colors_avg_category),
                  row=2, col=1)

    # Total Sales by Month
    total_sales_month = data.groupby("Month")["Sales"].sum().reset_index()
    colors_total_month = ['orange' if month in selected_months else 'lightgray' for month in total_sales_month['Month']]
    fig.add_trace(go.Bar(x=total_sales_month['Month'],
                         y=total_sales_month['Sales'],
                         marker_color=colors_total_month),
                  row=2, col=2)

    # Combined Bar Chart for Sales by State and Product Category
    combined_sales = data.groupby(["State", "Product_Category"])["Sales"].sum().reset_index()
    combined_sales['Selected'] = combined_sales.apply(
        lambda x: x['State'] == selected_state and x['Product_Category'] == selected_product_category, axis=1)
    colors_combined = np.where(combined_sales['Selected'], 'orange', 'blue')
    fig.add_trace(go.Bar(x=combined_sales['State'],
                         y=combined_sales['Sales'],
                         name='Sales',
                         marker_color=colors_combined),
                  row=2, col=3)

    # Update layout
    fig.update_layout(title_text="Sales Data Overview", showlegend=False, height=600, width=1200)
    fig.update_yaxes(tickprefix="₹")  # Add ₹ prefix to y-axis ticks
    st.plotly_chart(fig, use_container_width=True)

# Function to predict sales based on user input
def predict_sales(model, encoder, scaler, state, product_category, month):
    input_data = pd.DataFrame({
        "State": [state],
        "Product_Category": [product_category],
        "Month": [month]
    })

    # One-hot encode the input data
    input_encoded = encoder.transform(input_data[['State', 'Product_Category', 'Month']])

    # Scale the input data
    input_scaled = scaler.transform(input_encoded)

    # Predicting using the decision tree regression model
    log_sales_prediction = model.predict(input_scaled)
    sales_prediction = np.exp(log_sales_prediction[0]) - 1  # Adjusted to handle exponential transformation
    return sales_prediction

# Load model, encoder, and scaler
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')  # Load the model
    encoder = joblib.load('encoder.pkl')  # Load the encoder
    scaler = joblib.load('scaler.pkl')  # Load the scaler
    return model, encoder, scaler

# Streamlit UI
st.set_page_config(page_title="Sales Prediction", layout="wide")

st.title("Region Wise Product Category Sales Prediction")
st.markdown("### Upload your sales data and predict future sales based on state, product category, and month.")

# Load the model, encoder, and scaler
model, encoder, scaler = load_model()

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Load and preprocess data
    data = load_and_preprocess_data(uploaded_file)

    # Initialize session state for user inputs if not already done
    if 'state' not in st.session_state:
        st.session_state.state = data["State"].unique()[0]
    if 'product_category' not in st.session_state:
        st.session_state.product_category = data["Product_Category"].unique()[0]
    if 'month' not in st.session_state:
        st.session_state.month = data["Month"].unique()[0]
    if 'selected_state' not in st.session_state:
        st.session_state.selected_state = data["State"].unique()[0]
    if 'selected_product_category' not in st.session_state:
        st.session_state.selected_product_category = data["Product_Category"].unique()[0]
    if 'selected_months' not in st.session_state:
        st.session_state.selected_months = []

    # User input for predictions
    st.header("Predict Sales")
    state = st.selectbox("Select State", data["State"].unique(), index=0)
    product_category = st.selectbox("Select Product Category", data["Product_Category"].unique(), index=0)
    month = st.selectbox("Select Month", data["Month"].unique(), index=0)

    # When the user clicks the 'Predict' button
    if st.button("Predict"):
        sales_prediction = predict_sales(model, encoder, scaler, state, product_category, month)
        st.subheader("Predicted Sales:")
        st.write(f"Estimated Sales: ₹{sales_prediction:.2f}")

    # User input for selected attributes
    selected_state = st.selectbox("Highlight State", data["State"].unique(), index=0)
    selected_product_category = st.selectbox("Highlight Product Category", data["Product_Category"].unique(), index=0)
    selected_months = st.multiselect("Highlight Months", options=data["Month"].unique(), default=[])

    # Create charts with selected attributes highlighted
    create_charts(data, selected_state, selected_product_category, selected_months)

    # Reset button to clear inputs
    if st.button("Reset Inputs"):
        st.session_state.state = data["State"].unique()[0]
        st.session_state.product_category = data["Product_Category"].unique()[0]
        st.session_state.month = data["Month"].unique()[0]
        st.session_state.selected_state = data["State"].unique()[0]
        st.session_state.selected_product_category = data["Product_Category"].unique()[0]
        st.session_state.selected_months = []
