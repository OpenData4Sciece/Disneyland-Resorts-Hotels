# Disney Resorts' Hotels Analysis & Price Prediction

> A comprehensive analysis and price prediction system for Disney Resorts hotels worldwide

## Overview
This project combines traditional data analysis with machine learning to analyze Disney resort hotels globally. It features real-time data scraping, price prediction using PyTorch, and comprehensive statistical analysis. The system can analyze pricing strategies, occupancy patterns, customer satisfaction, and provide future price predictions.

## Project Structure
```
disney-resorts-hotels/
│
├── data-analysis-disney-hotels.py     # Original analysis script
├── disney_hotel_ml_predictor.py       # ML-based price prediction script
├── requirements.txt                   # Project dependencies
├── .gitignore                        # Git ignore file
├── license.md                        # License information
│
├── data/                             # Data directory
│   ├── raw/                          # Raw scraped data
│   └── processed/                    # Processed datasets
│
├── models/                           # Saved ML models
│   └── trained_models/              # Trained model checkpoints
│
├── tests/                           # Test directory
│   └── test_scraper.py             # Scraper tests
│   └── test_predictor.py           # Predictor tests
│
└── logs/                            # Logging directory
    └── scraping_logs/              # Web scraping logs
```

## Requirements
- Python 3.8+
- Required packages:
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=0.24.0
beautifulsoup4>=4.9.3
requests>=2.25.1
pytorch>=2.0.0
selenium>=4.1.0
webdriver_manager>=3.8.0
python-dateutil>=2.8.2
tqdm>=4.65.0
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/disney-resorts-hotels.git
   cd disney-resorts-hotels
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
The project offers two main scripts:

1. Traditional Analysis:
```bash
python data-analysis-disney-hotels.py
```

2. ML-based Price Prediction:
```bash
python disney_hotel_ml_predictor.py
```

## Features

### Real-time Data Collection
- Automated web scraping of Disney resort websites
- Real-time price and availability data
- Historical data collection and storage
- Rate limiting and error handling

### Data Analysis
- Statistical analysis of hotel metrics
- Price trend analysis
- Occupancy patterns
- Seasonal variations
- Special event impact analysis
- Customer satisfaction correlation

### Machine Learning Capabilities
- Neural network-based price prediction
- Feature engineering for ML models
- Real-time price forecasting
- Model performance evaluation
- GPU acceleration support
- Automated model retraining

### Visualization & Reporting
- Interactive visualizations
- Price trend charts
- Occupancy heat maps
- Satisfaction correlation matrices
- Performance metrics dashboards

## Customization
The system can be customized through various configuration options:

1. Scraping Parameters:
```python
scraper_config = {
    'delay': 2,  # Seconds between requests
    'max_retries': 3,
    'timeout': 30
}
```

2. Model Parameters:
```python
model_config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'hidden_layers': [64, 32]
}
```

## Output
The system generates:
- Processed datasets in CSV format
- Trained ML models
- Detailed analysis reports
- Visualization plots
- Performance metrics
- Log files

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the Apache-2.0 license - see the LICENSE.md file for details.

## Future Improvements
- Implementation of advanced ML architectures
- Addition of more Disney resorts
- Integration with booking systems
- Real-time price alerts
- Mobile app development
- API endpoint creation
- Enhanced visualization options

## Troubleshooting
Common issues and solutions:
- Scraping errors: Check network connection and update selectors
- Model training issues: Verify GPU availability and memory usage
- Data processing errors: Validate input data format and completeness

## Support
For questions and support:
- Open an issue on GitHub
- Check existing documentation
- Review closed issues for solutions

## Acknowledgments
- Disney resorts data architecture
- PyTorch community
- Web scraping best practices
- Hospitality industry insights
