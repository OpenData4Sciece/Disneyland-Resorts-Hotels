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

class GlobalDisneyAnalysis:
    def __init__(self):
        # Define all Disney resorts worldwide
        self.resorts = {
            'Disneyland Resort (California)': {
                'location': 'Anaheim, California, USA',
                'opening_year': 1955,
                'parks': ['Disneyland Park', 'Disney California Adventure'],
                'annual_visitors': 18700000,  # Pre-pandemic average
                'total_attractions': 89,
                'hotels': {
                    'Disney Grand Californian Hotel & Spa': {
                        'category': 'Deluxe',
                        'rooms': 948,
                        'base_price': 755,
                        'rating': 4.7,
                        'dining_venues': 6,
                        'spa': True,
                        'distance_to_park': 0.1,
                        'year_renovated': 2021,
                        'avg_satisfaction': 9.2
                    },
                    'Disneyland Hotel': {
                        'category': 'Deluxe',
                        'rooms': 973,
                        'base_price': 645,
                        'rating': 4.6,
                        'dining_venues': 4,
                        'spa': True,
                        'distance_to_park': 0.3,
                        'year_renovated': 2022,
                        'avg_satisfaction': 9.0
                    },
                    'Paradise Pier Hotel': {
                        'category': 'Moderate',
                        'rooms': 481,
                        'base_price': 445,
                        'rating': 4.2,
                        'dining_venues': 2,
                        'spa': False,
                        'distance_to_park': 0.4,
                        'year_renovated': 2020,
                        'avg_satisfaction': 8.5
                    }
                }
            },
            'Walt Disney World Resort (Florida)': {
                'location': 'Orlando, Florida, USA',
                'opening_year': 1971,
                'parks': ['Magic Kingdom', 'Epcot', 'Disney Hollywood Studios', 'Disney Animal Kingdom'],
                'annual_visitors': 58000000,
                'total_attractions': 172,
                'hotels': {
                    'Grand Floridian Resort & Spa': {
                        'category': 'Deluxe',
                        'rooms': 867,
                        'base_price': 890,
                        'rating': 4.8,
                        'dining_venues': 8,
                        'spa': True,
                        'distance_to_park': 0.2,
                        'year_renovated': 2022,
                        'avg_satisfaction': 9.4
                    },
                    'Contemporary Resort': {
                        'category': 'Deluxe',
                        'rooms': 655,
                        'base_price': 765,
                        'rating': 4.6,
                        'dining_venues': 6,
                        'spa': True,
                        'distance_to_park': 0.1,
                        'year_renovated': 2021,
                        'avg_satisfaction': 9.1
                    },
                    'Animal Kingdom Lodge': {
                        'category': 'Deluxe',
                        'rooms': 1293,
                        'base_price': 685,
                        'rating': 4.7,
                        'dining_venues': 4,
                        'spa': True,
                        'distance_to_park': 1.0,
                        'year_renovated': 2019,
                        'avg_satisfaction': 9.3
                    },
                    'Port Orleans Resort': {
                        'category': 'Moderate',
                        'rooms': 2048,
                        'base_price': 385,
                        'rating': 4.4,
                        'dining_venues': 3,
                        'spa': False,
                        'distance_to_park': 2.5,
                        'year_renovated': 2018,
                        'avg_satisfaction': 8.7
                    }
                }
            },
            'Tokyo Disney Resort': {
                'location': 'Tokyo, Japan',
                'opening_year': 1983,
                'parks': ['Tokyo Disneyland', 'Tokyo DisneySea'],
                'annual_visitors': 28700000,
                'total_attractions': 124,
                'hotels': {
                    'Disney Hotel MiraCosta': {
                        'category': 'Deluxe',
                        'rooms': 502,
                        'base_price': 795,
                        'rating': 4.8,
                        'dining_venues': 5,
                        'spa': True,
                        'distance_to_park': 0.1,
                        'year_renovated': 2020,
                        'avg_satisfaction': 9.5
                    },
                    'Tokyo Disneyland Hotel': {
                        'category': 'Deluxe',
                        'rooms': 706,
                        'base_price': 675,
                        'rating': 4.7,
                        'dining_venues': 4,
                        'spa': True,
                        'distance_to_park': 0.2,
                        'year_renovated': 2019,
                        'avg_satisfaction': 9.2
                    }
                }
            },
            'Disneyland Paris': {
                'location': 'Marne-la-Vallée, France',
                'opening_year': 1992,
                'parks': ['Disneyland Park', 'Walt Disney Studios Park'],
                'annual_visitors': 14900000,
                'total_attractions': 87,
                'hotels': {
                    'Disneyland Hotel': {
                        'category': 'Deluxe',
                        'rooms': 496,
                        'base_price': 800,
                        'rating': 4.8,
                        'dining_venues': 4,
                        'spa': True,
                        'distance_to_park': 0.1,
                        'year_renovated': 2023,
                        'avg_satisfaction': 9.3
                    },
                    'Disney Hotel New York': {
                        'category': 'Deluxe',
                        'rooms': 565,
                        'base_price': 500,
                        'rating': 4.5,
                        'dining_venues': 3,
                        'spa': True,
                        'distance_to_park': 0.5,
                        'year_renovated': 2021,
                        'avg_satisfaction': 8.9
                    },
                    'Newport Bay Club': {
                        'category': 'Moderate',
                        'rooms': 1098,
                        'base_price': 400,
                        'rating': 4.2,
                        'dining_venues': 2,
                        'spa': False,
                        'distance_to_park': 0.7,
                        'year_renovated': 2016,
                        'avg_satisfaction': 8.6
                    }
                }
            },
            'Hong Kong Disneyland Resort': {
                'location': 'Lantau Island, Hong Kong',
                'opening_year': 2005,
                'parks': ['Hong Kong Disneyland'],
                'annual_visitors': 6500000,
                'total_attractions': 34,
                'hotels': {
                    'Hong Kong Disneyland Hotel': {
                        'category': 'Deluxe',
                        'rooms': 400,
                        'base_price': 550,
                        'rating': 4.6,
                        'dining_venues': 3,
                        'spa': True,
                        'distance_to_park': 0.3,
                        'year_renovated': 2017,
                        'avg_satisfaction': 9.0
                    },
                    'Disney Explorer\'s Lodge': {
                        'category': 'Moderate',
                        'rooms': 750,
                        'base_price': 400,
                        'rating': 4.4,
                        'dining_venues': 2,
                        'spa': False,
                        'distance_to_park': 0.5,
                        'year_renovated': 2017,
                        'avg_satisfaction': 8.8
                    }
                }
            },
            'Shanghai Disney Resort': {
                'location': 'Shanghai, China',
                'opening_year': 2016,
                'parks': ['Shanghai Disneyland'],
                'annual_visitors': 11200000,
                'total_attractions': 42,
                'hotels': {
                    'Shanghai Disneyland Hotel': {
                        'category': 'Deluxe',
                        'rooms': 420,
                        'base_price': 475,
                        'rating': 4.7,
                        'dining_venues': 4,
                        'spa': True,
                        'distance_to_park': 0.2,
                        'year_renovated': 2020,
                        'avg_satisfaction': 9.1
                    },
                    'Toy Story Hotel': {
                        'category': 'Moderate',
                        'rooms': 800,
                        'base_price': 280,
                        'rating': 4.3,
                        'dining_venues': 2,
                        'spa': False,
                        'distance_to_park': 0.6,
                        'year_renovated': 2019,
                        'avg_satisfaction': 8.7
                    }
                }
            }
        }

    def generate_daily_data(self, start_date='2024-01-01', end_date='2024-12-31'):
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = []

        # Special events calendar (global Disney events)
        special_events = {
            'New Year': ('01-01', 1.4),
            'Lunar New Year': ('02-10', 1.3),  # Date varies by year
            'Valentine': ('02-14', 1.2),
            'Spring Break': ('03-15', 1.35),
            'Easter': ('03-31', 1.3),
            'Golden Week': ('05-01', 1.4),  # Japanese holiday
            'Summer Start': ('07-01', 1.4),
            'Halloween Season': ('10-01', 1.25),
            'Christmas Season': ('12-01', 1.35),
            'New Year Eve': ('12-31', 1.5)
        }

        for resort_name, resort_info in self.resorts.items():
            for hotel_name, hotel_info in resort_info['hotels'].items():
                for date in dates:
                    # Base seasonal factor
                    month = date.month
                    seasonal_factor = 1 + 0.3 * np.sin((month - 1) * np.pi / 6)
                    
                    # Regional adjustments
                    if 'Tokyo' in resort_name and date.month in [3, 4]:  # Cherry blossom season
                        seasonal_factor *= 1.2
                    elif 'Hong Kong' in resort_name and date.month in [7, 8]:  # Summer holiday
                        seasonal_factor *= 1.15
                    elif 'Paris' in resort_name and date.month in [6, 7, 8]:  # European summer
                        seasonal_factor *= 1.25
                    
                    # Special events check
                    event_factor = 1.0
                    for event, (event_date, factor) in special_events.items():
                        event_date = pd.to_datetime(f"2024-{event_date}")
                        if abs((date - event_date).days) <= 3:
                            event_factor = max(event_factor, factor)
                    
                    # Calculate occupancy
                    base_occupancy = np.random.normal(0.75, 0.1)
                    occupancy = min(1, max(0.3, base_occupancy * seasonal_factor * event_factor))
                    
                    # Calculate price with all factors
                    price = (hotel_info['base_price'] * 
                           seasonal_factor * 
                           event_factor * 
                           np.random.normal(1, 0.05))
                    
                    # Calculate bookings and revenue
                    bookings = int(occupancy * hotel_info['rooms'])
                    revenue = price * bookings
                    
                    # Generate satisfaction score (weighted by hotel's average satisfaction)
                    satisfaction = min(10, max(6, np.random.normal(
                        hotel_info['avg_satisfaction'], 
                        0.5
                    )))
                    
                    data.append({
                        'date': date,
                        'resort': resort_name,
                        'hotel': hotel_name,
                        'category': hotel_info['category'],
                        'rooms': hotel_info['rooms'],
                        'price': price,
                        'occupancy': occupancy,
                        'bookings': bookings,
                        'revenue': revenue,
                        'satisfaction': satisfaction,
                        'dining_venues': hotel_info['dining_venues'],
                        'spa': hotel_info['spa'],
                        'distance_to_park': hotel_info['distance_to_park'],
                        'year_renovated': hotel_info['year_renovated'],
                        'resort_annual_visitors': resort_info['annual_visitors'],
                        'resort_total_attractions': resort_info['total_attractions'],
                        'resort_opening_year': resort_info['opening_year']
                    })
        
        return pd.DataFrame(data)

    def analyze_global_patterns(self, df):
        # Resort-level analysis
        resort_stats = df.groupby('resort').agg({
            'revenue': ['sum', 'mean'],
            'occupancy': 'mean',
            'satisfaction': 'mean',
            'price': 'mean'
        }).round(2)

        # Hotel category analysis
        category_stats = df.groupby(['resort', 'category']).agg({
            'revenue': 'sum',
            'satisfaction': 'mean',
            'occupancy': 'mean'
        }).round(2)

        # Seasonal patterns by region
        seasonal = df.groupby([df['date'].dt.month, 'resort'])['occupancy'].mean().unstack()

        # Price-distance correlation
        price_distance_corr = df.groupby('resort').apply(
            lambda x: x['price'].corr(x['distance_to_park'])
        )

        # Satisfaction drivers analysis
        satisfaction_corr = df.groupby('resort').apply(
            lambda x: x[['satisfaction', 'price', 'dining_venues', 'distance_to_park']].corr()['satisfaction']
        )

        return {
            'resort_stats': resort_stats,
            'category_stats': category_stats,
            'seasonal_patterns': seasonal,
            'price_distance_corr': price_distance_corr,
            'satisfaction_drivers': satisfaction_corr
        }

    def create_visualizations(self, df):
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 15))

        # 1. Global Revenue Comparison
        plt.subplot(2, 2, 1)
        resort_revenue = df.groupby('resort')['revenue'].sum() / 1_000_000
        resort_revenue.sort_values(ascending=True).plot(kind='barh')
        plt.title('Total Revenue by Resort (Millions)')
        plt.xlabel('Revenue (Millions)')

        # 2. Satisfaction vs Price by Resort
        plt.subplot(2, 2, 2)
        for resort in df['resort'].unique():
            resort_data = df[df['resort'] == resort]
            plt.scatter(resort_data['price'], 
                       resort_data['satisfaction'],
                       alpha=0.5, 
                       label=resort)
        plt.title('Satisfaction vs Price by Resort')
        plt.xlabel('Price')
        plt.ylabel('Satisfaction Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 3. Seasonal Patterns
        plt.subplot(2, 2, 3)
        seasonal = df.groupby([df['date'].dt.month, 'resort'])['occupancy'].mean().unstack()
        seasonal.plot(marker='o')
        plt.title('Seasonal Occupancy Patterns by Resort')
        plt.xlabel('Month')
        plt.ylabel('Average Occupancy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 4. Hotel Categories Distribution
        plt.subplot(2, 2, 4)
        category_dist = df.groupby(['resort', 'category']).size().unstack()
        category_dist.plot(kind='bar', stacked=True)
        plt.title('Hotel Categories Distribution by Resort')
        plt.xlabel('Resort')
        plt.ylabel('Number of Hotels')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)

        plt.tight_layout()
        return fig

    def perform_advanced_analysis(self, df):
        """Perform advanced statistical analysis and modeling"""
        results = {}

        # 1. Price Prediction Model
        def create_price_model(data):
            X = data[['rooms', 'dining_venues', 'distance_to_park', 
                     'resort_annual_visitors', 'resort_total_attractions']]
            y = data['price']
            
            X = pd.get_dummies(X, columns=[])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            return {
                'r2_score': r2_score(y_test, y_pred),
                'feature_importance': dict(zip(X.columns, model.coef_))
            }

        results['price_models'] = {
            resort: create_price_model(resort_data)
            for resort, resort_data in df.groupby('resort')
        }

        # 2. Customer Satisfaction Analysis
        results['satisfaction_analysis'] = {
            'global_avg': df['satisfaction'].mean(),
            'by_category': df.groupby('category')['satisfaction'].mean(),
            'by_resort': df.groupby('resort')['satisfaction'].mean(),
            'correlation_matrix': df[['satisfaction', 'price', 'occupancy', 
                                    'dining_venues', 'distance_to_park']].corr()
        }

        # 3. Revenue Optimization Analysis
        def analyze_revenue_patterns(data):
            return {
                'peak_revenue_month': data.groupby(data['date'].dt.month)['revenue'].mean().idxmax(),
                'optimal_occupancy': data.groupby('occupancy').agg({
                    'revenue': 'mean',
                    'satisfaction': 'mean'
                }).sort_values('revenue', ascending=False).head(1),
                'price_elasticity': np.corrcoef(data['price'], data['bookings'])[0,1]
            }

        results['revenue_analysis'] = {
            resort: analyze_revenue_patterns(resort_data)
            for resort, resort_data in df.groupby('resort')
        }

        # 4. Competitive Analysis
        results['competitive_analysis'] = {
            'market_share': df.groupby('resort')['revenue'].sum() / df['revenue'].sum(),
            'avg_daily_rate': df.groupby('resort')['price'].mean(),
            'efficiency': df.groupby('resort').apply(
                lambda x: (x['revenue'].sum() / x['rooms'].iloc[0]).round(2)
            )
        }

        return results

    def generate_report(self, df, analysis_results):
        """Generate a comprehensive analysis report"""
        report = []
        
        # 1. Executive Summary
        report.append("=== Disney Global Resorts Analysis Report ===\n")
        report.append(f"Analysis Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        report.append(f"Total Resorts Analyzed: {df['resort'].nunique()}")
        report.append(f"Total Hotels Analyzed: {df['hotel'].nunique()}")
        
        # 2. Global Performance Metrics
        report.append("\n=== Global Performance Metrics ===")
        total_revenue = df['revenue'].sum() / 1_000_000
        avg_satisfaction = df['satisfaction'].mean()
        avg_occupancy = df['occupancy'].mean() * 100
        
        report.append(f"Total Revenue: ${total_revenue:.2f}M")
        report.append(f"Average Satisfaction Score: {avg_satisfaction:.2f}/10")
        report.append(f"Average Occupancy Rate: {avg_occupancy:.1f}%")
        
        # 3. Resort-Specific Analysis
        report.append("\n=== Resort-Specific Performance ===")
        for resort in df['resort'].unique():
            resort_data = df[df['resort'] == resort]
            report.append(f"\n{resort}:")
            report.append(f"- Revenue: ${resort_data['revenue'].sum()/1_000_000:.2f}M")
            report.append(f"- Satisfaction: {resort_data['satisfaction'].mean():.2f}/10")
            report.append(f"- Occupancy: {resort_data['occupancy'].mean()*100:.1f}%")
            report.append(f"- Price Prediction R²: {analysis_results['price_models'][resort]['r2_score']:.3f}")
        
        # 4. Key Findings
        report.append("\n=== Key Findings ===")
        
        # Revenue Leaders
        top_revenue_resort = df.groupby('resort')['revenue'].sum().idxmax()
        report.append(f"Top Revenue Generator: {top_revenue_resort}")
        
        # Satisfaction Leaders
        top_satisfaction_resort = df.groupby('resort')['satisfaction'].mean().idxmax()
        report.append(f"Highest Customer Satisfaction: {top_satisfaction_resort}")
        
        # Efficiency Analysis
        revenue_per_room = df.groupby('resort').apply(
            lambda x: (x['revenue'].sum() / x['rooms'].iloc[0])
        ).sort_values(ascending=False)
        report.append(f"Most Efficient Resort (Revenue/Room): {revenue_per_room.index[0]}")
        
        # 5. Recommendations
        report.append("\n=== Recommendations ===")
        
        # Price Optimization
        price_elastic_resorts = {
            resort: data['price_elasticity']
            for resort, data in analysis_results['revenue_analysis'].items()
        }
        sensitive_resort = min(price_elastic_resorts.items(), key=lambda x: x[1])[0]
        report.append(f"Price Sensitivity: {sensitive_resort} shows highest price sensitivity")
        
        return "\n".join(report)

def main():
    # Initialize analyzer
    analyzer = GlobalDisneyAnalysis()
    
    # Generate data
    df = analyzer.generate_daily_data()
    
    # Perform analyses
    global_patterns = analyzer.analyze_global_patterns(df)
    advanced_analysis = analyzer.perform_advanced_analysis(df)
    
    # Create visualizations
    visualizations = analyzer.create_visualizations(df)
    
    # Generate report
    report = analyzer.generate_report(df, advanced_analysis)
    
    # Print report
    print(report)
    
    # Save data and visualizations
    df.to_csv('disney_global_analysis.csv', index=False)
    visualizations.savefig('disney_analysis_visualizations.png')

if __name__ == "__main__":
    main()
