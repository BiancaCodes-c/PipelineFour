import pandas as pd
import numpy as np

class SalesDataProcessor:
    def __init__(self, filepath):
        self.raw_data = pd.read_csv(filepath)
        self.processed_data = None

    def clean_data(self):
        df = self.raw_data.copy()

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Remove rows with missing important values
        df = df.dropna(subset=['date','quantity','unit_price','customer_age'])

        # Remove negative or zero values
        df = df[(df['quantity'] > 0) & (df['unit_price'] > 0)]

        # Remove unrealistic ages
        df = df[(df['customer_age'] > 0) & (df['customer_age'] < 100)]

        # Create revenue column
        df['revenue'] = df['quantity'] * df['unit_price']

        self.processed_data = df
        return self

    def create_time_features(self):

        if self.processed_data is None:
            self.clean_data()

        df = self.processed_data

        # Extract time features
        df['day_of_week'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter

        # Weekend flag
        df['weekend'] = df['date'].dt.dayofweek >= 5

        self.processed_data = df
        return self

    def segment_customers(self):

        df = self.processed_data

        # Age bins
        bins = [0,18,25,35,50,100]

        labels = [
            "Under 18",
            "18-25",
            "26-35",
            "36-50",
            "Over 50"
        ]

        df['age_group'] = pd.cut(df['customer_age'], bins=bins, labels=labels)

        self.processed_data = df
        return self

    def calculate_metrics(self):

        df = self.processed_data

        # Daily total sales
        daily_sales = df.groupby('date')['revenue'].sum().reset_index()

        # Weekly revenue by store
        df['week'] = df['date'].dt.isocalendar().week
        weekly_store = df.groupby(['week','store_id'])['revenue'].sum().reset_index()

        # Average quantity by product and age group
        product_age = df.groupby(['product_id','age_group'])['quantity'].mean().reset_index()

        # Revenue trends by month and quarter
        monthly_trend = df.groupby('month')['revenue'].sum().reset_index()
        quarterly_trend = df.groupby('quarter')['revenue'].sum().reset_index()

        # Save metrics
        self.metrics = {
            "daily_sales": daily_sales,
            "weekly_store": weekly_store,
            "product_age": product_age,
            "monthly_trend": monthly_trend,
            "quarterly_trend": quarterly_trend
        }

        return self

    def export_files(self):

        df = self.processed_data

        # Save processed dataset
        df.to_csv("processed_sales_data.csv", index=False)

        # Save metrics
        self.metrics['daily_sales'].to_csv("daily_sales.csv", index=False)
        self.metrics['weekly_store'].to_csv("weekly_store_revenue.csv", index=False)
        self.metrics['product_age'].to_csv("product_age_quantity.csv", index=False)
        self.metrics['monthly_trend'].to_csv("monthly_revenue.csv", index=False)
        self.metrics['quarterly_trend'].to_csv("quarterly_revenue.csv", index=False)

        return self

    def process(self):

        return (self.clean_data()
                    .create_time_features()
                    .segment_customers()
                    .calculate_metrics()
                    .export_files()
                    .processed_data)


# Run pipeline
processor = SalesDataProcessor('sales_data.csv')
processed_data = processor.process()

print(processed_data.head())
