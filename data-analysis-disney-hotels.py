import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class DisneyHotelAnalysis:
    def __init__(self):
        self.hotels = {
            'Disneyland Hotel': {
                'base_price': 800,
                'seasonality': 1.5,
                'rating': 4.8,
                'capacity': 496,
                'stars': 5,
                'distance_to_park': 0.1,
                'renovation_year': 2023,
                'restaurants': 4,
                'pool': True,
                'spa': True,
                'category': 'Luxury'
            },
            'Disney Hotel New York': {
                'base_price': 500,
                'seasonality': 1.4,
                'rating': 4.5,
                'capacity': 565,
                'stars': 4,
                'distance_to_park': 0.5,
                'renovation_year': 2021,
                'restaurants': 3,
                'pool': True,
                'spa': True,
                'category': 'Deluxe'
            },
            'Newport Bay Club': {
                'base_price': 400,
                'seasonality': 1.3,
                'rating': 4.2,
                'capacity': 1098,
                'stars': 4,
                'distance_to_park': 0.7,
                'renovation_year': 2016,
                'restaurants': 2,
                'pool': True,
                'spa': False,
                'category': 'Moderate'
            },
            'Sequoia Lodge': {
                'base_price': 350,
                'seasonality': 1.2,
                'rating': 4.0,
                'capacity': 1011,
                'stars': 3,
                'distance_to_park': 0.8,
                'renovation_year': 2012,
                'restaurants': 2,
                'pool': True,
                'spa': False,
                'category': 'Moderate'
            },
            'Hotel Cheyenne': {
                'base_price': 250,
                'seasonality': 1.1,
                'rating': 3.8,
                'capacity': 1000,
                'stars': 3,
                'distance_to_park': 1.2,
                'renovation_year': 2017,
                'restaurants': 1,
                'pool': False,
                'spa': False,
                'category': 'Value'
            },
            'Hotel Santa Fe': {
                'base_price': 200,
                'seasonality': 1.1,
                'rating': 3.6,
                'capacity': 1000,
                'stars': 2,
                'distance_to_park': 1.5,
                'renovation_year': 2012,
                'restaurants': 1,
                'pool': False,
                'spa': False,
                'category': 'Value'
            }
        }
        
    def generate_data(self):
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        data = []
        
        # Special events calendar
        special_events = {
            'Valentine': ('2024-02-14', 1.2),
            'Easter': ('2024-04-01', 1.3),
            'Summer Start': ('2024-07-01', 1.4),
            'Halloween': ('2024-10-31', 1.25),
            'Christmas': ('2024-12-25', 1.35),
            'New Year': ('2024-12-31', 1.4)
        }

        for date in dates:
            # Base seasonal factor
            month = date.month
            seasonal_factor = 1 + 0.3 * np.sin((month - 1) * np.pi / 6)
            
            # Check for special events
            event_factor = 1.0
            for event, (event_date, factor) in special_events.items():
                event_date = pd.to_datetime(event_date)
                if abs((date - event_date).days) <= 3:  # Effect spans 3 days before and after
                    event_factor = max(event_factor, factor)
            
            # Weather effect (simplified)
            weather_factor = 1 + 0.1 * np.random.randn()
            
            for hotel, details in self.hotels.items():
                # Calculate occupancy
                base_occupancy = np.random.normal(0.75, 0.1)
                occupancy = min(1, max(0.3, base_occupancy * seasonal_factor * event_factor * weather_factor))
                
                # Calculate price with all factors
                price = (details['base_price'] * 
                        seasonal_factor * 
                        event_factor * 
                        weather_factor)
                
                # Add random variation
                price *= np.random.normal(1, 0.05)
                
                # Generate customer satisfaction scores
                satisfaction = min(5, max(1, np.random.normal(details['rating'], 0.5)))
                
                # Generate number of bookings
                bookings = int(occupancy * details['capacity'])
                
                # Calculate revenue
                revenue = price * bookings
                
                # Generate customer demographics (simplified)
                families_pct = np.random.normal(0.6, 0.1)
                couples_pct = np.random.normal(0.3, 0.1)
                solo_pct = 1 - families_pct - couples_pct
                
                data.append({
                    'date': date,
                    'hotel': hotel,
                    'price': price,
                    'occupancy': occupancy,
                    'bookings': bookings,
                    'revenue': revenue,
                    'satisfaction': satisfaction,
                    'families_pct': families_pct,
                    'couples_pct': couples_pct,
                    'solo_pct': solo_pct,
                    'stars': details['stars'],
                    'distance_to_park': details['distance_to_park'],
                    'renovation_year': details['renovation_year'],
                    'restaurants': details['restaurants'],
                    'pool': details['pool'],
                    'spa': details['spa'],
                    'category': details['category']
                })
        
        return pd.DataFrame(data)

    def perform_analysis(self, df):
        # 1. Basic Statistical Analysis
        print("\n=== Basic Statistical Analysis ===")
        stats_summary = df.groupby('hotel').agg({
            'price': ['mean', 'std', 'min', 'max'],
            'occupancy': ['mean', 'std'],
            'revenue': ['mean', 'sum'],
            'satisfaction': ['mean', 'std']
        }).round(2)
        print(stats_summary)
        
        # 2. Correlation Analysis
        numeric_cols = ['price', 'occupancy', 'satisfaction', 'revenue', 'stars', 
                       'distance_to_park', 'restaurants']
        correlation_matrix = df[numeric_cols].corr()
        
        # 3. Time Series Analysis
        monthly_revenue = df.groupby(['hotel', pd.Grouper(key='date', freq='M')])['revenue'].sum().unstack(0)
        
        # 4. Customer Segment Analysis
        segment_analysis = df.groupby('hotel').agg({
            'families_pct': 'mean',
            'couples_pct': 'mean',
            'solo_pct': 'mean'
        }).round(3)
        
        # 5. Price Elasticity Analysis
        def calculate_price_elasticity(hotel_data):
            price_pct_change = hotel_data['price'].pct_change()
            demand_pct_change = hotel_data['bookings'].pct_change()
            elasticity = (demand_pct_change / price_pct_change).mean()
            return elasticity
        
        elasticities = df.groupby('hotel').apply(calculate_price_elasticity)
        
        # 6. Machine Learning: Price Prediction Model
        X = df[['stars', 'distance_to_park', 'restaurants', 'occupancy']]
        y = df['price']
        
        X = pd.get_dummies(X, columns=[])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        # Visualization
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Revenue Trends
        plt.subplot(2, 2, 1)
        monthly_revenue.plot(marker='o')
        plt.title('Monthly Revenue by Hotel')
        plt.xlabel('Date')
        plt.ylabel('Revenue (€)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Price vs. Satisfaction Scatter
        plt.subplot(2, 2, 2)
        for hotel in self.hotels.keys():
            hotel_data = df[df['hotel'] == hotel]
            plt.scatter(hotel_data['price'], hotel_data['satisfaction'], 
                       alpha=0.5, label=hotel)
        plt.title('Price vs. Customer Satisfaction')
        plt.xlabel('Price (€)')
        plt.ylabel('Satisfaction Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Customer Segment Distribution
        plt.subplot(2, 2, 3)
        segment_analysis.plot(kind='bar', stacked=True)
        plt.title('Customer Segment Distribution by Hotel')
        plt.xlabel('Hotel')
        plt.ylabel('Percentage')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Correlation Heatmap
        plt.subplot(2, 2, 4)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Key Metrics')
        
        plt.tight_layout()
        
        return {
            'stats_summary': stats_summary,
            'correlation_matrix': correlation_matrix,
            'monthly_revenue': monthly_revenue,
            'segment_analysis': segment_analysis,
            'elasticities': elasticities,
            'price_prediction_r2': r2
        }

# Execute the analysis
analysis = DisneyHotelAnalysis()
df = analysis.generate_data()
results = analysis.perform_analysis(df)

# Print key findings
print("\n=== Key Findings ===")
print(f"\nPrice Prediction Model R² Score: {results['price_prediction_r2']:.3f}")
print("\nPrice Elasticity by Hotel:")
print(results['elasticities'].round(3))

# Additional insights
high_value_hotels = df.groupby('hotel')['revenue'].sum().sort_values(ascending=False)
print("\nHotels Ranked by Total Revenue:")
print(high_value_hotels.round(2))

# Perform statistical tests
f_stat, p_value = stats.f_oneway(*[group['satisfaction'].values 
                                  for name, group in df.groupby('hotel')])
print("\nANOVA Test for Satisfaction Across Hotels:")
print(f"F-statistic: {f_stat:.3f}")
print(f"p-value: {p_value:.3f}")

# Save results to CSV
df.to_csv('disney_hotels_analysis.csv', index=False)
print("\nFull dataset saved to 'disney_hotels_analysis.csv'")
