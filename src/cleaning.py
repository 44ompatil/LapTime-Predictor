import numpy as np 
import pandas as pd 
import os

class DataCleaning:
    def __init__(self):
        try:
            self.df = pd.read_parquet(path=r"data\raw\rawData.parquet")
        except Exception as e:
            print(e)

    def dataOverview(self):
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

    def clean(self):
        #Already have the driver initials
        self.df.drop(columns=['DriverName'], inplace=True)

        #DeltaTimes to seconds
        self.df['LapTimeSeconds'] = self.df['LapTime'].dt.total_seconds()
        self.df['Sector1TimeSeconds'] = self.df['Sector1Time'].dt.total_seconds()
        self.df['Sector2TimeSeconds'] = self.df['Sector2Time'].dt.total_seconds()
        self.df['Sector3TimeSeconds'] = self.df['Sector3Time'].dt.total_seconds()

        #Dropping null values, it consists of invalid laps, outlaps and inLaps
        self.df.dropna(subset=['Sector3TimeSeconds','Sector3Time'], inplace=True)
        self.df.dropna(subset=['Sector2TimeSeconds','Sector2Time'], inplace=True)
        self.df.dropna(subset=['Sector1TimeSeconds','Sector1Time'], inplace=True)
        self.df.dropna(subset=['LapTimeSeconds','LapTime'], inplace=True)
        self.df.dropna(subset=['TyreLife'], inplace=True)

        #Constructing inlap and outlap
        self.df['OutLap'] = self.df['PitOutTime'].notna()
        self.df['InLap'] = self.df['PitInTime'].notna()

        #Dropping laps when the car has entered or exited the pit lane
        self.df.drop(self.df[self.df['OutLap'] == True].index, inplace=True)
        self.df.drop(self.df[self.df['InLap'] == True].index, inplace=True)

        self.df.drop(columns=['OutLap', 'InLap'], inplace=True)

        #Delete unwanted columns
        self.df.drop(columns=['PitInTime','PitOutTime','LapTime','Sector1Time','Sector2Time','Sector3Time'], inplace=True)

        self.df.drop_duplicates(inplace=True)

        #Save
        os.makedirs(r"data\processed", exist_ok=True)
        self.df.to_parquet(r"data\processed\cleanedData.parquet", index=False)
        self.df.to_csv(r"data\processed\cleanedData.csv", index=False)
        print("Cleaned data saved.")


if __name__ == "__main__":
    cleaning = DataCleaning()
    cleaning.dataOverview()
    cleaning.clean()
    print("Data Cleaning completed successfully")