import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# Create sample data
np.random.seed(42)

# Generate dates for the last year
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

hotels = {
    'Disney Hotel New York': {
        'base_price': 500,
        'seasonality': 1.4,
        'rating': 4.5,
        'capacity': 565
    },
    'Disneyland Hotel': {
        'base_price': 800,
        'seasonality': 1.5,
        'rating': 4.8,
        'capacity': 496
    },
    'Newport Bay Club': {
        'base_price': 400,
        'seasonality': 1.3,
        'rating': 4.2,
        'capacity': 1098
    },
    'Sequoia Lodge': {
        'base_price': 350,
        'seasonality': 1.2,
        'rating': 4.0,
        'capacity': 1011
    },
}

# Create DataFrame
data = []
for date in dates:
    # Seasonal factor (higher in summer and during holidays)
    month = date.month
    seasonal_factor = 1 + 0.3 * np.sin((month - 1) * np.pi / 6)  # Peak in July
    
    # Holiday premium (Christmas, New Year, Easter)
    holiday_premium = 1.0
    if (date.month == 12 and date.day >= 20) or (date.month == 1 and date.day <= 5):
        holiday_premium = 1.3
    
    for hotel, details in hotels.items():
        # Calculate occupancy (random with seasonal influence)
        base_occupancy = np.random.normal(0.75, 0.1)
        occupancy = min(1, max(0.3, base_occupancy * seasonal_factor * holiday_premium))
        
        # Calculate price with seasonality
        price = details['base_price'] * seasonal_factor * holiday_premium
        
        # Add some random variation
        price *= np.random.normal(1, 0.05)
        
        data.append({
            'date': date,
            'hotel': hotel,
            'price': price,
            'occupancy': occupancy,
            'rating': details['rating'],
            'capacity': details['capacity']
        })

df = pd.DataFrame(data)

# Create visualizations
plt.style.use('seaborn')
fig = plt.figure(figsize=(20, 15))

# 1. Average Price by Hotel
plt.subplot(2, 2, 1)
avg_prices = df.groupby('hotel')['price'].mean().sort_values(ascending=True)
avg_prices.plot(kind='barh')
plt.title('Average Price by Hotel')
plt.xlabel('Price (€)')
plt.ylabel('Hotel')

# 2. Occupancy Rate Over Time
plt.subplot(2, 2, 2)
for hotel in hotels.keys():
    hotel_data = df[df['hotel'] == hotel]
    plt.plot(hotel_data['date'], hotel_data['occupancy'], label=hotel, alpha=0.7)
plt.title('Occupancy Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Occupancy Rate')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tick_params(axis='x', rotation=45)

# 3. Price vs. Rating Scatter Plot
plt.subplot(2, 2, 3)
plt.scatter(df.groupby('hotel')['rating'].mean(), 
           df.groupby('hotel')['price'].mean(),
           s=df.groupby('hotel')['capacity'].mean() / 5)
for hotel in hotels.keys():
    plt.annotate(hotel, 
                (df[df['hotel'] == hotel]['rating'].mean(), 
                 df[df['hotel'] == hotel]['price'].mean()))
plt.title('Price vs. Rating (bubble size represents capacity)')
plt.xlabel('Rating')
plt.ylabel('Average Price (€)')

# 4. Monthly Average Price Trends
plt.subplot(2, 2, 4)
monthly_avg = df.groupby(['hotel', df['date'].dt.month])['price'].mean().unstack()
monthly_avg.plot(marker='o')
plt.title('Monthly Average Price Trends')
plt.xlabel('Month')
plt.ylabel('Average Price (€)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Calculate some key statistics
stats = pd.DataFrame({
    'Average Price': df.groupby('hotel')['price'].mean(),
    'Average Occupancy': df.groupby('hotel')['occupancy'].mean() * 100,
    'Rating': df.groupby('hotel')['rating'].first(),
    'Capacity': df.groupby('hotel')['capacity'].first(),
    'Revenue Potential': df.groupby('hotel').apply(lambda x: x['price'].mean() * x['capacity'].iloc[0] * x['occupancy'].mean())
}).round(2)

print("\nHotel Statistics:")
print(stats)
