"""
Disney Hotel Price Predictor with ML
This script scrapes real-time Disney hotel data and uses PyTorch to predict hotel prices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import logging
import json

class DisneyHotelScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            filename='disney_scraper.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def setup_driver(self):
        """Initialize Selenium WebDriver with Chrome"""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    def scrape_disney_hotels(self):
        """Scrape hotel data from Disney resort websites"""
        resorts_data = {}
        resort_urls = {
            'Disneyland_Resort': 'https://disneyland.disney.go.com/hotels/',
            'Walt_Disney_World': 'https://disneyworld.disney.go.com/resorts/',
            'Disneyland_Paris': 'https://www.disneylandparis.com/en-gb/hotels/',
            # Add other Disney resort URLs
        }

        driver = self.setup_driver()
        
        try:
            for resort_name, url in resort_urls.items():
                logging.info(f"Scraping data for {resort_name}")
                resorts_data[resort_name] = self._scrape_resort(driver, url, resort_name)
                time.sleep(2)  # Polite delay between requests
        
        except Exception as e:
            logging.error(f"Error during scraping: {str(e)}")
        
        finally:
            driver.quit()
            
        return resorts_data

    def _scrape_resort(self, driver, url, resort_name):
        """Scrape individual resort data"""
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        hotels = []
        hotel_elements = driver.find_elements(By.CSS_SELECTOR, '.hotel-listing')
        
        for hotel in hotel_elements:
            try:
                hotel_data = {
                    'name': hotel.find_element(By.CSS_SELECTOR, '.hotel-name').text,
                    'category': self._extract_category(hotel),
                    'price': self._extract_price(hotel),
                    'rating': self._extract_rating(hotel),
                    'amenities': self._extract_amenities(hotel),
                    'location': self._extract_location(hotel)
                }
                hotels.append(hotel_data)
                
            except Exception as e:
                logging.error(f"Error scraping hotel in {resort_name}: {str(e)}")
                
        return hotels

    def _extract_category(self, hotel_element):
        """Extract hotel category"""
        try:
            category_element = hotel_element.find_element(By.CSS_SELECTOR, '.hotel-category')
            return category_element.text
        except:
            return "Not specified"

    def _extract_price(self, hotel_element):
        """Extract hotel price"""
        try:
            price_element = hotel_element.find_element(By.CSS_SELECTOR, '.price-value')
            price_text = price_element.text.replace('$', '').replace(',', '')
            return float(price_text)
        except:
            return None

    def _extract_rating(self, hotel_element):
        """Extract hotel rating"""
        try:
            rating_element = hotel_element.find_element(By.CSS_SELECTOR, '.rating-value')
            return float(rating_element.text)
        except:
            return None

    def _extract_amenities(self, hotel_element):
        """Extract hotel amenities"""
        try:
            amenities_elements = hotel_element.find_elements(By.CSS_SELECTOR, '.amenity')
            return [amenity.text for amenity in amenities_elements]
        except:
            return []

    def _extract_location(self, hotel_element):
        """Extract hotel location"""
        try:
            location_element = hotel_element.find_element(By.CSS_SELECTOR, '.location')
            return location_element.text
        except:
            return "Location not specified"

class DisneyHotelDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class DisneyPricePredictionModel(nn.Module):
    def __init__(self, input_size):
        super(DisneyPricePredictionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

class GlobalDisneyAnalysis:
    def __init__(self):
        self.scraper = DisneyHotelScraper()
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self):
        """Prepare data for analysis and modeling"""
        # Scrape current hotel data
        raw_data = self.scraper.scrape_disney_hotels()
        
        # Convert to DataFrame and preprocess
        df = self._preprocess_data(raw_data)
        
        # Generate additional features
        df = self._generate_features(df)
        
        return df

    def _preprocess_data(self, raw_data):
        """Preprocess scraped data"""
        processed_data = []
        
        for resort, hotels in raw_data.items():
            for hotel in hotels:
                hotel_data = {
                    'resort': resort,
                    'name': hotel['name'],
                    'category': hotel['category'],
                    'price': hotel['price'],
                    'rating': hotel['rating'],
                    'amenities_count': len(hotel['amenities']),
                    'has_spa': 'spa' in [a.lower() for a in hotel['amenities']],
                    'has_pool': 'pool' in [a.lower() for a in hotel['amenities']],
                    'has_restaurant': 'restaurant' in [a.lower() for a in hotel['amenities']]
                }
                processed_data.append(hotel_data)
        
        return pd.DataFrame(processed_data)

    def _generate_features(self, df):
        """Generate additional features for analysis"""
        # Convert categorical variables to dummy variables
        df = pd.get_dummies(df, columns=['resort', 'category'])
        
        # Add temporal features
        df['day_of_week'] = datetime.now().weekday()
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_holiday'] = self._check_holidays(datetime.now()).astype(int)
        
        return df

    def _check_holidays(self, date):
        """Check if date is during major holidays or peak seasons"""
        # Add holiday checking logic here
        return False

    def train_model(self, df):
        """Train PyTorch model for price prediction"""
        # Prepare features and target
        features = df.drop(['price', 'name'], axis=1, errors='ignore')
        targets = df['price']
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, targets, test_size=0.2, random_state=42
        )
        
        # Create datasets and dataloaders
        train_dataset = DisneyHotelDataset(X_train, y_train)
        test_dataset = DisneyHotelDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Initialize model
        model = DisneyPricePredictionModel(input_size=features.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        epochs = 100
        for epoch in tqdm(range(epochs), desc="Training model"):
            model.train()
            for batch_features, batch_targets in train_loader:
                batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs.squeeze(), batch_targets)
                loss.backward()
                optimizer.step()
        
        return model, (X_test, y_test)

    def evaluate_model(self, model, test_data):
        """Evaluate trained model"""
        X_test, y_test = test_data
        model.eval()
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions = model(X_test_tensor).cpu().numpy().squeeze()
            
            mse = np.mean((predictions - y_test) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - y_test))
            
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

    def generate_predictions(self, model, new_data):
        """Generate predictions for new data"""
        model.eval()
        scaled_data = self.scaler.transform(new_data)
        
        with torch.no_grad():
            predictions = model(torch.FloatTensor(scaled_data).to(self.device))
            return predictions.cpu().numpy().squeeze()

def main():
    # Initialize analyzer
    analyzer = GlobalDisneyAnalysis()
    
    try:
        # Prepare data
        print("Scraping and preparing data...")
        df = analyzer.prepare_data()
        
        # Train model
        print("Training model...")
        model, test_data = analyzer.train_model(df)
        
        # Evaluate model
        print("Evaluating model...")
        evaluation_results = analyzer.evaluate_model(model, test_data)
        
        # Print results
        print("\nModel Evaluation Results:")
        print(f"Root Mean Square Error: ${evaluation_results['rmse']:.2f}")
        print(f"Mean Absolute Error: ${evaluation_results['mae']:.2f}")
        
        # Save model and data
        torch.save(model.state_dict(), 'disney_hotel_model.pth')
        df.to_csv('disney_hotel_data.csv', index=False)
        
        print("\nModel and data saved successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
