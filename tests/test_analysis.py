import pytest
import pandas as pd
import numpy as np
from disney_hotels_analysis import DisneyHotelAnalysis

@pytest.fixture
def analysis():
    return DisneyHotelAnalysis()

@pytest.fixture
def sample_data(analysis):
    return analysis.generate_data()

def test_data_generation(analysis):
    df = analysis.generate_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert all(col in df.columns for col in ['date', 'hotel', 'price', 'occupancy'])

def test_data_validity(sample_data):
    # Test price ranges
    assert all(sample_data['price'] > 0)
    
    # Test occupancy ranges
    assert all(sample_data['occupancy'].between(0, 1))
    
    # Test satisfaction ranges
    assert all(sample_data['satisfaction'].between(1, 5))

def test_analysis_results(analysis, sample_data):
    results = analysis.perform_analysis(sample_data)
    
    # Test if all expected results are present
    expected_keys = ['stats_summary', 'correlation_matrix', 'monthly_revenue',
                    'segment_analysis', 'elasticities', 'price_prediction_r2']
    assert all(key in results for key in expected_keys)
    
    # Test R² score is between 0 and 1
    assert 0 <= results['price_prediction_r2'] <= 1

def test_seasonal_patterns(sample_data):
    # Test if summer months have higher prices
    summer_prices = sample_data[sample_data['date'].dt.month.isin([6, 7, 8])]['price']
    winter_prices = sample_data[sample_data['date'].dt.month.isin([12, 1, 2])]['price']
    assert summer_prices.mean() > winter_prices.mean()
