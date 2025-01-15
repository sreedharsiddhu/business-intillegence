import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot

# Load data
file_meteo_csv = r"C:\Desktop\power bi\new data sets\dati_meteo_storici_(Cicalino1).csv"
file_catture_csv = r"C:\Desktop\power bi\new data sets\grafico_delle_catture_(Cicalino 1).csv"

meteo_df = pd.read_csv(file_meteo_csv, sep=',', skiprows=2)
catture_df = pd.read_csv(file_catture_csv, sep=',', skiprows=2)

# Rename columns
meteo_df.columns = ['DateTime', 'Media_Temperatura', 'Low_Temp', 'High_Temp', 'Media_Umidita']
catture_df.columns = ['DateTime', 'Numero_Insetti', 'Nuove_Catture', 'Recensito', 'Evento']

# Data cleaning and conversion
meteo_df['Media_Temperatura'] = meteo_df['Media_Temperatura'].astype(str).str.replace(',', '.', regex=False).astype(float)
catture_df['Numero_Insetti'] = catture_df['Numero_Insetti'].fillna(0).astype(int)
catture_df['Nuove_Catture'] = catture_df['Nuove_Catture'].fillna(0).astype(int)

# Convert datetime columns
meteo_df['DateTime'] = pd.to_datetime(meteo_df['DateTime'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
catture_df['DateTime'] = pd.to_datetime(catture_df['DateTime'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

meteo_df['Date'] = meteo_df['DateTime'].dt.date
catture_df['Date'] = catture_df['DateTime'].dt.date

# Filter captures for 6:00 AM
catture_filtered = catture_df[catture_df['DateTime'].dt.hour == 6][['Date', 'Numero_Insetti', 'Nuove_Catture']]

# Merge datasets
meteo_with_catture = pd.merge(meteo_df, catture_filtered, on='Date', how='left')
meteo_with_catture['Numero_Insetti'] = meteo_with_catture['Numero_Insetti'].fillna(0).astype(int)
meteo_with_catture['Nuove_Catture'] = meteo_with_catture['Nuove_Catture'].fillna(0).astype(int)

# Streamlit layout
st.title('Meteorological Data and Insect Captures Analysis')

# Show data
st.subheader('Weather and Insect Data')
st.write(meteo_with_catture.head())

# Histogram of Temperature and Humidity
st.subheader('Temperature and Humidity Distribution')
col1, col2 = st.columns(2)

with col1:
    st.write('Temperature Histogram')
    fig1, ax1 = plt.subplots()
    meteo_with_catture['Media_Temperatura'].hist(bins=20, ax=ax1)
    st.pyplot(fig1)

with col2:
    st.write('Humidity Histogram')
    fig2, ax2 = plt.subplots()
    meteo_with_catture['Media_Umidita'].hist(bins=20, ax=ax2)
    st.pyplot(fig2)

# Autocorrelation plot of insect captures
st.subheader('Autocorrelation of Insect Captures')
daily_insects = meteo_with_catture.set_index('Date')['Numero_Insetti'].resample('D').sum()
fig3, ax3 = plt.subplots()
autocorrelation_plot(daily_insects)
plt.title('Autocorrelation of the Number of Insects')
st.pyplot(fig3)

# Seasonal decomposition of insect captures
st.subheader('Seasonal Decomposition of Insect Captures')
decomposition = seasonal_decompose(daily_insects, model='additive', period=7)
fig4 = decomposition.plot()
st.pyplot(fig4)

# Residual analysis
st.subheader('Residual Analysis')
residui = decomposition.resid
fig5, ax5 = plt.subplots(figsize=(12, 6))
ax5.plot(residui, marker='o', linestyle='-', color='b')
ax5.axhline(0, color='red', linestyle='--')
plt.title("Residuals of the Model")
plt.xlabel("Data")
plt.ylabel("Residuals")
st.pyplot(fig5)

# Polynomial regression for Temperature vs Insect Count
st.subheader('Polynomial Regression: Temperature vs Number of Insects')
poly = PolynomialFeatures(degree=2)  # Degree 2 model
X_poly = poly.fit_transform(meteo_with_catture[['Media_Temperatura']])
model = LinearRegression()
model.fit(X_poly, meteo_with_catture['Numero_Insetti'])
st.write("Score:", model.score(X_poly, meteo_with_catture['Numero_Insetti']))

# Display correlation
st.subheader('Correlation: Temperature vs Number of Insects')
correlation_temp = meteo_with_catture[['Media_Temperatura', 'Numero_Insetti']].corr()
st.write(correlation_temp)

# Density plot for Temperature vs Insect Count
st.subheader('Density: Temperature vs Number of Insects')
fig6, ax6 = plt.subplots()
sns.kdeplot(x=meteo_with_catture['Media_Temperatura'], y=meteo_with_catture['Numero_Insetti'], cmap="Blues", fill=True, ax=ax6)
plt.title('Density: Temperature vs Number of Insects')
plt.xlabel('Media_Temperatura')
plt.ylabel('Number of Insects')
st.pyplot(fig6)
