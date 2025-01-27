# Disneyland Resorts Hotels

Disneyland Resorts Hotels Investigation Study


# Disney Hotels Analysis Project

## Overview
This project performs a comprehensive data science analysis of Disneyland Paris hotels, including pricing strategies, occupancy patterns, customer satisfaction, and revenue optimization. The analysis includes statistical modeling, machine learning predictions, and detailed visualizations.

## Project Structure
```
Disneyland-Resorts-Hotels/
│
├── disney_hotels_analysis.py    # Main analysis script
├── requirements.txt            # Project dependencies
├── data/                      # Data directory
│   └── disney_hotels_analysis.csv  # Generated dataset
│
├── notebooks/                 # Jupyter notebooks
│   └── analysis.ipynb        # Interactive analysis
│
└── visualizations/           # Generated plots and charts
```

## Requirements
- Python 3.8+
- Required packages:
  ```
  numpy==1.21.0
  pandas==1.3.0
  matplotlib==3.4.2
  seaborn==0.11.1
  scikit-learn==0.24.2
  scipy==1.7.0
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/OpenData4Sciece/Disneyland-Resorts-Hotels.git
   cd Disneyland-Resorts-Hotels
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
1. Run the main analysis script:
   ```bash
   python python disney_hotels_analysis.py
   ```

2. Or import the DisneyHotelAnalysis class in your own script:
   ```python
   from disney_hotels_analysis import DisneyHotelAnalysis
   
   analysis = DisneyHotelAnalysis()
   df = analysis.generate_data()
   results = analysis.perform_analysis(df)
   ```

## Features

### Data Generation
- Simulates realistic hotel data including:
  - Daily prices and occupancy rates
  - Seasonal variations
  - Special event effects
  - Weather impacts
  - Customer demographics

### Analysis Components
1. **Basic Statistical Analysis**
   - Mean, standard deviation, min/max for key metrics
   - Hotel-wise comparisons
   - Temporal patterns

2. **Advanced Analytics**
   - Price elasticity calculation
   - Customer segment analysis
   - Revenue optimization
   - Satisfaction analysis

3. **Machine Learning**
   - Price prediction model
   - Occupancy forecasting
   - Customer satisfaction modeling

4. **Visualizations**
   - Revenue trends
   - Price-satisfaction relationships
   - Customer segment distribution
   - Correlation heatmaps

## Customization
The analysis can be customized by modifying the following parameters in the `DisneyHotelAnalysis` class:

1. Hotel Properties:
   ```python
   self.hotels = {
       'hotel_name': {
           'base_price': float,
           'seasonality': float,
           'rating': float,
           # ... other properties
       }
   }
   ```

2. Special Events:
   ```python
   special_events = {
       'event_name': ('date', factor)
   }
   ```

## Output
The script generates:
1. Detailed statistical analysis in the console
2. Visualization plots
3. CSV file with the complete dataset
4. Machine learning model performance metrics

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the Apache-2.0 license - see the LICENSE file for details.

## Acknowledgments
- Inspired by real-world hotel data analysis
- Uses best practices from hospitality industry analytics
- Implements standard data science methodologies

## Contact
For questions and feedback, please open an issue on the GitHub.

## Future Improvements
- Add more sophisticated machine learning models
- Implement real-time data processing
- Create interactive dashboard
- Add more granular customer segmentation
- Integrate with external weather APIs
- Expand visualization options

## References
- Hotel Industry Statistical Analysis Methods
- Python Data Science Best Practices
- Machine Learning for Hospitality Industry
