import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler

# Connect to SQLite database (or your preferred SQL database)
conn = sqlite3.connect('sensor_data.db')

# Fetch data using SQL
query = """
SELECT timestamp, vibration, temperature
FROM sensor_readings
WHERE timestamp BETWEEN '2023-01-01' AND '2023-12-31'
"""
data = pd.read_sql_query(query, conn)

# Display basic information
print(data.info())
print(data.describe())

# Preprocess the data
data = data.dropna()

# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['vibration', 'temperature']])

# After anomaly detection, store results back in the database
def store_anomalies(timestamps, anomaly_flags):
    cursor = conn.cursor()
    cursor.executemany("""
    INSERT INTO anomalies (timestamp, is_anomaly)
    VALUES (?, ?)
    """, zip(timestamps, anomaly_flags))
    conn.commit()

# Don't forget to close the connection when done
conn.close()

