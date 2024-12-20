import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import urllib


def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

try:
    df_temp = load_data("https://github.com/sreedharsiddhu/business-intillegence/blob/main/Temperature.csv")
    df_insect = load_data("https://github.com/sreedharsiddhu/business-intillegence/blob/main/Insect_Caught.csv")


except urllib.error.URLError as e:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    df_temp = load_data("https://github.com/sreedharsiddhu/business-intillegence/blob/main/Temperature.csv")
    df_insect = load_data("https://github.com/sreedharsiddhu/business-intillegence/blob/main/Insect_Caught.csv")


# Drop duplicates and null values
def data_cleaning(data):
    data = data.drop_duplicates()
    data = data.dropna()
    return data

df_insect = data_cleaning(df_insect)
df_temp = data_cleaning(df_temp)

def date_parsing(df):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')  # Uses dayfirst and handles errors
    return df

df_insect = date_parsing(df_insect)
df_temp = date_parsing(df_temp)

# Group by date and take the mean of the temperature values
temp_cols = ['Mean_Temperature', 'Temperature_Low', 'Temperature_High', 'Mean_Humidity']

def convert_float(df, cols):
    for col in temp_cols:
        df[col] = df[col].str.replace(',', '.').astype(float)
    return df


df_temp = convert_float(df_temp, temp_cols)
df_temp = df_temp.groupby('Date').mean().reset_index()
df_insect = df_insect.groupby('Date').sum().reset_index()

df_merged = pd.merge(df_insect, df_temp, on='Date', how='inner')


df_merged['Prev_Num_Insects'] = df_merged['Number_of_Insects'].shift(1)
df_merged['Prev_Temperature'] = df_merged['Mean_Temperature'].shift(1)
df_merged['Prev_Humidity'] = df_merged['Mean_Humidity'].shift(1)
df_merged['Temp_Delta'] = df_merged['Temperature_High'] - df_merged['Temperature_Low']
df_merged['Rolling_Temperature'] = df_merged['Mean_Temperature'].rolling(window=3).mean()
df_merged['Rolling_Humidity'] = df_merged['Mean_Humidity'].rolling(window=3).mean()

df_cleaned = df_merged.dropna(subset=['Mean_Temperature', 'Temperature_Low', 'Temperature_High',
                                      'Mean_Humidity', 'Prev_Num_Insects', 'Prev_Temperature',
                                      'Prev_Humidity', 'Temp_Delta', 'Rolling_Temperature', 
                                      'Rolling_Humidity', 'Number_of_Insects', 'New_Catches'])
                                

st.sidebar.title("Navigation")
menu = ["About Us","Visualization","Correlation", "Hypothesis Testing", "Model Evaluation", "Inference"]
page = st.sidebar.selectbox("Choose an option", menu)
models = ["Regression", "Classification"]

@st.cache_data
def plot_correlation(df, cols: list[str]):
    correlation_matrix = df[cols].corr()

    # Create a Plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='Greys',
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )

    # Update layout
    fig.update_layout(width=800, height=700)
    return fig, correlation_matrix

if page == menu[0]:
    st.title(menu[0])
    st.subheader("Santosh kumar Yadav")
    st.write("Matricola Number: D03000048")
    st.subheader("Syed Najam Mehdi")
    st.write("Matricola Number: D03000049")
    st.subheader("Pujan Thapa")
    st.write("Matricola Number: D03000046")

    
if page == menu[1]:
    st.title(menu[1])
    if st.button("No. of Insects"):
        df_daily = df_merged.groupby('Date').agg({
            'Number_of_Insects': 'sum',
            'New_Catches': 'sum',
            'Mean_Temperature': 'mean',
            'Mean_Humidity': 'mean'
        }).reset_index()

        # Create an interactive Plotly line chart
        fig = px.line(
            df_daily,
            x='Date',
            y='Number_of_Insects',
            title="Number of Insects Caught - Daily Aggregation",
            labels={'Number_of_Insects': 'Number of Insects', 'Date': 'Date'},
            line_shape='spline',  # Smoother line
            markers=True  # Add markers to data points
        )

        fig.update_layout(
            title={'x': 0.5},  # Center the title
            xaxis_title="Date",
            yaxis_title="Number of Insects",
            xaxis=dict(showgrid=True, tickangle=45),
            yaxis=dict(showgrid=True),
            template="plotly_white"  # Use a clean theme
        )

        st.plotly_chart(fig, use_container_width=True)
        st.write("Most of the insects have been caught in the month of August.")

    if st.button("No. of Catches"):
        # Aggregate daily data
        df_daily = df_merged.groupby('Date').agg({
            'Number_of_Insects': 'sum',
            'New_Catches': 'sum',
            'Mean_Temperature': 'mean',
            'Mean_Humidity': 'mean'
        }).reset_index()

        fig_new_catches = px.line(
            df_daily,
            x='Date',
            y='New_Catches',
            title="New Insect Catches - Daily Aggregation",
            labels={'New_Catches': 'New Catches', 'Date': 'Date'},
            line_shape='spline',
            markers=True
        )
        fig_new_catches.update_layout(
            title={'x': 0.5},
            xaxis_title="Date",
            yaxis_title="New Catches",
            xaxis=dict(showgrid=True, tickangle=45),
            yaxis=dict(showgrid=True),
            template="plotly_white"
        )
        st.plotly_chart(fig_new_catches, use_container_width=True)
        st.write("Month seems to have no connection with the number of new catches.")

    if st.button("Distribution"):
        st.subheader("Distribution of New Insect Catches")
        # Create the Plotly histogram
        fig = px.histogram(
            df_merged, 
            x='New_Catches', 
            nbins=20,  # Number of bins
            opacity=0.7
        )
        fig.update_traces(histnorm='density')  # Normalize histogram to show density
        fig.update_layout(
            xaxis_title="Number of New Catches",
            yaxis_title="Frequency",
            template="plotly_white"  # Optional: set a Plotly theme
        )

        # Render the plot in Streamlit
        st.plotly_chart(fig)
        st.write("The distribution of new insect catches is right-skewed.")

elif page == menu[2]:
    st.title("Correlation Analysis")
    st.subheader("Correlation Matrix Between Pest Counts and Weather Variables")
    cols = ["Number_of_Insects", "Mean_Temperature", "Mean_Humidity"]
    fig, corr_mat = plot_correlation(df_merged, cols)
    st.plotly_chart(fig)
    st.write("Correlation Matrix:")
    st.dataframe(corr_mat)
    st.write("There appears to be a positive albeit weak correlation between the number of insects and the mean humidity. However, there is no significant linear relationship between Number of Insects and Mean Temperature.")

elif page == menu[3]:
    st.title(menu[3])

    variable = st.selectbox("Choose Variable for Hypothesis Testing", ["Temperature", "Humidity"])

    if variable == "Temperature":
        st.write("### Hypothesis: Temperature significantly affects the number of insects caught.")
        df_cleaned['Temp_Category'] = pd.cut(df_cleaned['Mean_Temperature'], bins=3, labels=['Low', 'Medium', 'High'])

        low_temp = df_cleaned[df_cleaned['Temp_Category'] == 'Low']['Number_of_Insects']
        high_temp = df_cleaned[df_cleaned['Temp_Category'] == 'High']['Number_of_Insects']

        t_stat, p_val = stats.ttest_ind(low_temp, high_temp, equal_var=False)

        st.write("### Results")
        st.write(f"T-Statistic: {t_stat:.2f}")
        st.write(f"P-Value: {p_val:.4f}")

        if p_val < 0.05:
            st.write("The result is statistically significant. Temperature has a significant effect on the number of insects caught.")
        else:
            st.write("The result is not statistically significant. Temperature does not have a significant effect on the number of insects caught.")

    elif variable == "Humidity":
        st.write("### Hypothesis: Humidity significantly affects the number of insects caught.")
        df_cleaned['Humidity_Category'] = pd.cut(df_cleaned['Mean_Humidity'], bins=3, labels=['Low', 'Medium', 'High'])

        low_humidity = df_cleaned[df_cleaned['Humidity_Category'] == 'Low']['Number_of_Insects']
        high_humidity = df_cleaned[df_cleaned['Humidity_Category'] == 'High']['Number_of_Insects']

        t_stat, p_val = stats.ttest_ind(low_humidity, high_humidity, equal_var=False)

        st.write("### Results")
        st.write(f"T-Statistic: {t_stat:.2f}")
        st.write(f"P-Value: {p_val:.4f}")

        if p_val < 0.05:
            st.write("The result is statistically significant. Humidity has a significant effect on the number of insects caught.")
        else:
            st.write("The result is not statistically significant. Humidity does not have a significant effect on the number of insects caught.")

elif page == menu[4]:
    st.title(menu[4])
    if st.button(models[0]):
        X_reg = df_cleaned[['Mean_Temperature', 'Temperature_Low', 'Temperature_High', 'Mean_Humidity',
                            'Prev_Num_Insects', 'Prev_Temperature', 'Prev_Humidity', 'Temp_Delta',
                            'Rolling_Temperature', 'Rolling_Humidity']]
        y_reg = df_cleaned['Number_of_Insects']
        
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.15, random_state=42)
        
        scaler = StandardScaler()
        X_train_reg_scaled = scaler.fit_transform(X_train_reg)
        X_test_reg_scaled = scaler.transform(X_test_reg)
        
        rf_reg = RandomForestRegressor()
        rf_reg.fit(X_train_reg_scaled, y_train_reg)
        y_pred_reg = rf_reg.predict(X_test_reg_scaled)
        
        rmse = mean_squared_error(y_test_reg, y_pred_reg, squared=False)
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        
        st.subheader("Regression Model Evaluation")
        st.write(f'RMSE: {rmse}')
        st.write(f'MAE: {mae}')
    
    if st.button(models[1]):
        X_clf = df_cleaned[['Mean_Temperature', 'Temperature_Low', 'Temperature_High', 'Mean_Humidity',
                            'Prev_Num_Insects', 'Prev_Temperature', 'Prev_Humidity', 'Temp_Delta',
                            'Rolling_Temperature', 'Rolling_Humidity']]
        y_clf = (df_cleaned['New_Catches'] > 0).astype(int)

        
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.15, random_state=42)

        
        scaler = StandardScaler()
        X_train_clf_scaled = scaler.fit_transform(X_train_clf)
        X_test_clf_scaled = scaler.transform(X_test_clf)
        
        rf_clf = RandomForestClassifier()
        rf_clf.fit(X_train_clf_scaled, y_train_clf)
        y_pred_clf = rf_clf.predict(X_test_clf_scaled)
        
        accuracy = accuracy_score(y_test_clf, y_pred_clf)
        
        st.subheader("Classification Model Evaluation")
        st.write(f'Accuracy: {accuracy}')

elif page == menu[5]:
    st.title(menu[5])
    model_type = st.radio("Select Model Type", models)
    st.subheader("Input Features")
    mean_temp = st.slider("Select a Mean Temperature (°C)", min_value=-10, max_value=100, value=25)
    temp_low = st.slider("Select a Low Temperature (°C)", min_value=-10, max_value=100, value=25)
    temp_high = st.slider("Select a High Temperature (°C)", min_value=-10, max_value=100, value=25)
    mean_humidity = st.slider("Select a Mean Humidity (%)", min_value=0, max_value=100, value=25)
    prev_temp = st.slider("Select a Previous Temperature (°C)", min_value=-10, max_value=100, value=25)
    prev_humidity = st.slider("Select a Previous Humidity (%)", min_value=0, max_value=100, value=25)
    rolling_temp = st.slider("Select a Rolling Mean Temperature", min_value=-10, max_value=100, value=25)
    rolling_humidity = st.slider("Select a Rolling Mean Humidity (%)", min_value=0, max_value=100, value=25)
    temp_delta = temp_high - temp_low
    prev_num_insects = st.number_input("Previous Number of Insects", value=0)

    input_features = np.array([[mean_temp, temp_low, temp_high, mean_humidity,
                                prev_num_insects, prev_temp, prev_humidity,
                                temp_delta, rolling_temp, rolling_humidity]])
    
    scaler = StandardScaler()
    df_features = df_cleaned[['Mean_Temperature', 'Temperature_Low', 'Temperature_High', 
                              'Mean_Humidity', 'Prev_Num_Insects', 'Prev_Temperature', 
                              'Prev_Humidity', 'Temp_Delta', 'Rolling_Temperature', 
                              'Rolling_Humidity']]
    scaler.fit(df_features)
    input_scaled = scaler.transform(input_features)
    
    if model_type == models[0]:
        X_reg = df_features
        y_reg = df_cleaned['Number_of_Insects']
        rf_reg = RandomForestRegressor()
        rf_reg.fit(X_reg, y_reg)
        prediction = rf_reg.predict(input_scaled)
        rounded_prediction = np.round(prediction, 0)
        st.write(f"Predicted Number of Insects: {rounded_prediction[0]:.1f}")
    
    elif model_type == models[1]:
        X_clf = df_features
        y_clf = (df_cleaned['New_Catches'] > 0).astype(int)
        rf_clf = RandomForestClassifier()
        rf_clf.fit(X_clf, y_clf)
        prediction = rf_clf.predict(input_scaled)
        prediction_prob = rf_clf.predict_proba(input_scaled)
        probabilities_df = pd.DataFrame(prediction_prob, columns=rf_clf.classes_)
        st.write("Prediction Probabilities:")
        st.dataframe(probabilities_df)
        max_probability = probabilities_df.max(axis=1)
        predicted_class = probabilities_df.idxmax(axis=1)
        st.write(f"Predicted Class: {predicted_class[0]} (0: No Catch, 1: Catch)")
        st.write(f"Maximum Probability: {max_probability[0]:.2%}")

