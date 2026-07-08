import os
import numpy as np 
import pandas as pd 
import fastf1
import matplotlib.pyplot as plt 
import seaborn as sns 

class plotResults:
    def __init__(self, data_path=None, output_dir=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        if data_path is None:
            self.data_path = os.path.join(project_root, 'data', 'processed', 'featEngineeredData.parquet')
        else:
            self.data_path = data_path
            
        if output_dir is None:
            self.output_dir = os.path.join(project_root, 'pics', 'plots')
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Loading data from: {self.data_path}")
        self.df = pd.read_parquet(self.data_path)
        print(f"Data loaded successfully. Shape: {self.df.shape}")

    def plot_tyre_life(self):
        print("Generating Tyre Life vs Lap Time scatter plot")
        plt.figure(figsize=(16, 5))
        plt.scatter(x=self.df['TyreLife'], y=self.df['LapTimeSeconds'], alpha=0.4)
        plt.xlabel("Tyre Life")
        plt.ylabel("Lap Time (s)")
        plt.title("Tyre Life vs Lap Time")
        save_path = os.path.join(self.output_dir, 'tyre_life.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_tyre_life_circuit(self):
        print("Generating Tyre Life vs Mean Lap Time per Circuit")
        circuits = self.df['Circuit'].unique() # Total 29 Unique Circuits

        fig, axes = plt.subplots(10, 3, figsize=(22, 50))
        axes = axes.flatten()

        for i, cir in enumerate(circuits):
            currDF = self.df[(self.df['Circuit'] == cir) & (self.df['RaceSession'] == "Race") & (self.df['newStint?'] == False)]

            if not currDF.empty:
                Q1 = currDF['LapTimeSeconds'].quantile(0.25)
                Q3 = currDF['LapTimeSeconds'].quantile(0.75)
                IQR = Q3 - Q1
                currDF = currDF[currDF['LapTimeSeconds'] <= Q3 + 1.5 * IQR]

                mean_lap = currDF.groupby('TyreLife')['LapTimeSeconds'].mean()
                axes[i].plot(mean_lap.index, mean_lap.values, color='steelblue')
            
            axes[i].set_title(cir, fontsize=9)
            axes[i].set_xlabel('Tyre Life')
            axes[i].set_ylabel('Avg Lap Time (s)')
            axes[i].grid(True, alpha=0.3)

        for j in range(len(circuits), len(axes)):
            axes[j].set_visible(False)

        plt.subplots_adjust(hspace=0.5, wspace=0.35)
        plt.suptitle('Tyre Life vs Mean Lap Time — Per Circuit', fontsize=14, fontweight='bold', y=1.001)
        save_path = os.path.join(self.output_dir, 'tyre_life_circuit.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_track_temp(self):
        print("Generating Track Temp vs Mean Lap Time per Circuit")
        circuits = self.df['Circuit'].unique()

        fig, axes = plt.subplots(10, 3, figsize=(22, 50))
        axes = axes.flatten()

        for i, cir in enumerate(circuits):
            currDF = self.df[(self.df['Circuit'] == cir) & (self.df['RaceSession'] == "Race") & (self.df['newStint?'] == False)]

            if not currDF.empty:
                Q1 = currDF['LapTimeSeconds'].quantile(0.25)
                Q3 = currDF['LapTimeSeconds'].quantile(0.75)
                IQR = Q3 - Q1
                currDF = currDF[currDF['LapTimeSeconds'] <= Q3 + 1.5 * IQR]

                currDF = currDF.copy() # Avoid SettingWithCopyWarning
                currDF['TrackTempConvt'] = currDF['TrackTemp'].round(0)
                mean_lap = currDF.groupby('TrackTempConvt')['LapTimeSeconds'].mean()
                axes[i].plot(mean_lap.index, mean_lap.values, color='steelblue')
            
            axes[i].set_title(cir, fontsize=9)
            axes[i].set_xlabel('Track Temp')
            axes[i].set_ylabel('Avg Lap Time')
            axes[i].grid(True, alpha=0.3)

        for j in range(len(circuits), len(axes)):
            axes[j].set_visible(False)

        plt.subplots_adjust(hspace=0.5, wspace=0.35)
        plt.suptitle('Track Temp vs Mean Lap Time — Per Circuit', fontsize=14, fontweight='bold', y=1.001)
        save_path = os.path.join(self.output_dir, 'track_temp.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_correlation_heatmap(self):
        print("Generating Feature Correlation Heatmap")
        race_df = self.df[(self.df['RaceSession'] == 'Race') & (self.df['newStint?'] == False)]
        
        if race_df.empty:
            print("No race session data available for correlation heatmap.")
            return

        Q1, Q3 = race_df['LapTimeSeconds'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        race_df = race_df[race_df['LapTimeSeconds'] <= Q3 + 1.5 * IQR]

        numericalCols = ['LapNumber', 'TyreLife', 'AirTemp', 'TrackTemp', 'Humidity',
                    'PrevLapTime', 'Prev2LapTime', 'Prev3LapTime',
                    'AvgPrev2', 'AvgPrev3', 'StdPrev5', 'LapTimeSeconds']
        
        
        existing_cols = [col for col in numericalCols if col in race_df.columns]
        corr = race_df[existing_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 9))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax, linewidths=0.5, annot_kws={'size': 8})
        ax.set_title('Feature Correlation Heatmap (Race Laps Only)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'corr.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_lap_number(self):
        print("Generating Lap Number vs Mean Lap Time per Circuit")
        race_df = self.df[(self.df['RaceSession'] == 'Race') & (self.df['newStint?'] == False)]

        if race_df.empty:
            print("No race session data available for lap number plot.")
            return

        circuits = race_df['Circuit'].unique()
        fig, axes = plt.subplots(10, 3, figsize=(22, 50))
        axes = axes.flatten()

        for i, cir in enumerate(circuits):
            currDF = race_df[race_df['Circuit'] == cir].copy()
            
            if not currDF.empty:
                Q1, Q3 = currDF['LapTimeSeconds'].quantile([0.25, 0.75])
                currDF = currDF[currDF['LapTimeSeconds'] <= Q3 + 1.5 * (Q3 - Q1)]

                mean_lap = currDF.groupby('LapNumber')['LapTimeSeconds'].mean()
                axes[i].plot(mean_lap.index, mean_lap.values, color='tomato')
            
            axes[i].set_title(cir, fontsize=9, fontweight='bold')
            axes[i].set_xlabel('Lap Number')
            axes[i].set_ylabel('Avg Lap Time (s)')
            axes[i].grid(True, alpha=0.3)

    
        for j in range(len(circuits), len(axes)):
            axes[j].set_visible(False)

        plt.subplots_adjust(hspace=0.5, wspace=0.35)
        plt.suptitle('Lap Number vs Mean Lap Time — Per Circuit', fontsize=14, fontweight='bold', y=1.001)
        save_path = os.path.join(self.output_dir, 'lap_num.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_weather_features(self):
        print("Generating Weather Features vs Lap Time plot")
        race_df = self.df[(self.df['RaceSession'] == 'Race') & (self.df['newStint?'] == False)]

        if race_df.empty:
            print("No race session data available for weather features plot.")
            return

        Q1, Q3 = race_df['LapTimeSeconds'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        race_df = race_df[race_df['LapTimeSeconds'] <= Q3 + 1.5 * IQR].copy()

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
    
        race_df['RoundedAirTemp'] = race_df['AirTemp'].round(0)
        air_mean = race_df.groupby('RoundedAirTemp')['LapTimeSeconds'].mean()
        axes[0].plot(air_mean.index, air_mean.values, color='darkorange', linewidth=2)
        axes[0].set_title('AirTemp vs Avg Lap Time', fontweight='bold')
        axes[0].set_xlabel('Air Temperature (°C)')
        axes[0].set_ylabel('Avg Lap Time (s)')
        axes[0].grid(True, alpha=0.3)

    
        race_df['RoundedHumidity'] = race_df['Humidity'].round(0)
        hum_mean = race_df.groupby('RoundedHumidity')['LapTimeSeconds'].mean()
        axes[1].plot(hum_mean.index, hum_mean.values, color='teal', linewidth=2)
        axes[1].set_title('Humidity vs Avg Lap Time', fontweight='bold')
        axes[1].set_xlabel('Humidity (%)')
        axes[1].set_ylabel('Avg Lap Time (s)')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Weather Features vs Lap Time', fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'weather.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_all(self):
        print("Starting batch plotting...")
        self.plot_tyre_life()
        self.plot_tyre_life_circuit()
        self.plot_track_temp()
        self.plot_correlation_heatmap()
        self.plot_lap_number()
        self.plot_weather_features()
        print("All plots generated successfully!")

if __name__ == "__main__":
    plotter = plotResults()
    plotter.plot_all()