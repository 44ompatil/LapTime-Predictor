from IPython.core import display_functions
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

class FeatureEngineering:
    def __init__(self):
        try:
            self.df = pd.read_parquet(path=r"data\processed\cleanedData.parquet")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = None

    def dataOverview(self):
        if self.df is None or self.df.empty:
            print("No data available to display overview.")
            return
        print("="*50)
        print(self.df.head())
        print("="*50)
        print(self.df.info())
        print("="*50)
        print(self.df.describe())
        print("="*50)
        print(self.df.dtypes)
        print("="*50)
        print(self.df.isnull().sum())
        print("="*50)

    def engineerFeatures(self):
        if self.df is None or self.df.empty:
            print("Dataframe is empty. Cannot engineer features.")
            return

        self.df = self.df.sort_values(by=["SeasonYear", "RaceName", "RaceSession", "Driver", "LapNumber"])

        groupBy = self.df.groupby(["SeasonYear", "RaceName", "RaceSession", "Driver"])

       
        self.df['PrevLapTime'] = groupBy["LapTimeSeconds"].shift(1)
        self.df['PrevTyreLife'] = groupBy['TyreLife'].shift(1)
        self.df['newStint?'] = self.df["TyreLife"] < self.df["PrevTyreLife"]
        self.df['StintID'] = groupBy['newStint?'].cumsum() + 1
        
        self.df['Prev2LapTime'] = groupBy['LapTimeSeconds'].shift(2)
        self.df['Prev3LapTime'] = groupBy['LapTimeSeconds'].shift(3)
        
    
        self.df['AvgPrev2'] = groupBy['LapTimeSeconds'].transform(lambda x: x.rolling(2).mean())
        self.df['AvgPrev3'] = groupBy['LapTimeSeconds'].transform(lambda x: x.rolling(3).mean())
        self.df['StdPrev5'] = groupBy['LapTimeSeconds'].transform(lambda x: x.rolling(5).mean())

  
        self.df.drop(self.df[self.df['Compound'] == 'None'].index, inplace=True)
        self.df.drop(self.df[self.df['Compound'] == 'UNKNOWN'].index, inplace=True)

    
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.set_output(transform='pandas')
        
        EncCompound = encoder.fit_transform(self.df[['Compound']])
        
        self.df = pd.concat([self.df.drop(columns=['Compound']), EncCompound], axis=1)

        os.makedirs(r"data\processed", exist_ok=True)
        self.df.to_parquet(r"data\processed\featEngineeredData.parquet", index=False)
        self.df.to_csv(r"data\processed\featEngineeredData.csv", index=False)
        print("Feature engineered data saved.")


if __name__ == "__main__":
    featEngg = FeatureEngineering()
    featEngg.dataOverview()
    featEngg.engineerFeatures()
    print("Feature Engineering completed successfully.")
