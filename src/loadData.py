import fastf1
import numpy as np
import pandas as pd
import pathlib

import basicConfig
import chores

class dataLoader:
    def __init__(self) -> None:
        bc = basicConfig.config()
        bc.enableCache()

    def loadSession(self, seasonYear: int, circuitName: str, raceSession: str):
        try:
            session = fastf1.get_session(seasonYear, circuitName, raceSession)
            return session
        except Exception as e:
            print(f"Error getting session: {e}")
            return None

    def loadCircuitData(self, session):
        try:
            meeting = session.session_info.get("Meeting") or {}
            circuit = meeting.get("Circuit") or {}
            
            raceName = meeting.get("Name", "")
            circuitName = circuit.get("ShortName", "")
            raceSession = session.session_info.get("Name", "")
            
            return [raceName, circuitName, raceSession]
        except Exception as e:
            print(f"Error loading circuit data: {e}")
            return None

    def loadDriverTeam(self, session, driverName: str):
        try:
            driver_info = session.get_driver(driverName)
            if driver_info is not None:
                fullName = driver_info.get("FullName", "")
                teamName = driver_info.get("TeamName", "")
                return [fullName, teamName]
            return ["", ""]
        except Exception as e:
            print(f"Error loading driver/team info for {driverName}: {e}")
            return ["", ""]

    def loadLapData(self, session):
        try:
            cols = [
                "Driver",
                "LapNumber",
                "LapTime",
                "Sector1Time",
                "Sector2Time",
                "Sector3Time",
                "TyreLife",
                "Compound",
                "PitOutTime",
                "PitInTime",
            ]
            laps = session.laps.copy()
            for col in cols:
                if col not in laps.columns:
                    laps[col] = np.nan
            return laps[cols].copy()
        except Exception as e:
            print(f"Error loading lap data: {e}")
            return None

    def loadWeatherData(self, session):
        try:
            weather = session.weather_data[["AirTemp", "TrackTemp", "Humidity"]]
            return weather
        except Exception as e:
            print(f"Error loading weather data: {e}")
            return None

    def loadData(self, seasonYear: int):
        data = pd.DataFrame()

        for circuit in chores.circuits:
            for session in chores.sessions:
                print(f"Loading data for {seasonYear} - {circuit} - {session}\n")
                currentSession = self.loadSession(seasonYear, circuit, session)
                if currentSession is None:
                    continue

                try:
                    currentSession.load(laps=True, telemetry=False, weather=True, messages=False)
                except Exception as e:
                    print(f"Failed to load session data: {e}")
                    continue

                circuitInfo = self.loadCircuitData(currentSession)
                if circuitInfo is None:
                    raceName, circuitName, raceSession = "", "", ""
                else:
                    raceName, circuitName, raceSession = circuitInfo

                laps = self.loadLapData(currentSession)
                if laps is None or laps.empty:
                    continue

                try:
                    weatherPerlap = currentSession.laps.get_weather_data()[["AirTemp", "TrackTemp", "Humidity"]]
                    weatherPerlap.index = laps.index
                except Exception as e:
                    print(f"Error getting weather data: {e}")
                    weatherPerlap = pd.DataFrame(np.nan, index=laps.index, columns=["AirTemp", "TrackTemp", "Humidity"])

                driverNames = {}
                driverTeams = {}
                for d in laps["Driver"].unique():
                    driver_info = self.loadDriverTeam(currentSession, d)
                    if driver_info:
                        driverNames[d] = driver_info[0]
                        driverTeams[d] = driver_info[1]

                laps["DriverName"] = laps["Driver"].map(driverNames)
                laps["TeamName"] = laps["Driver"].map(driverTeams)

                laps["RaceName"] = raceName
                laps["Circuit"] = circuitName
                laps["RaceSession"] = raceSession
                laps["SeasonYear"] = seasonYear

                session_df = pd.concat([laps, weatherPerlap], axis=1)
                data = pd.concat([data, session_df], ignore_index=True)

        return data


if __name__ == '__main__':
    dl = dataLoader()
    cfg = basicConfig.config()

    combinedData = []
    for year in chores.seasonYears:
        data = dl.loadData(year)
        if data is not None and not data.empty:
            combinedData.append(data)

    if combinedData:
        finalDF = pd.concat(combinedData, ignore_index=True)
        cfg.exportDatatoParquet(finalDF,pathlib.Path(r'D:\Project\LapTime-Predictor\data\raw\rawData.parquet'))
        finalDF.to_csv(r"D:\Project\LapTime-Predictor\data\raw\rawData.csv")
    else:
        print("No data was loaded.")