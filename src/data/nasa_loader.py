import numpy as np
import pandas as pd


class NASADataLoader:
    """
    Generates synthetic weather and load data mimicking NASA Natural Resources database
    characteristics as described in the paper (Figs 8-9).
    """

    def __init__(self, steps_per_episode=24):
        self.steps = steps_per_episode

    def generate_year_data(self, hours=8760):
        """
        Generates 1 year of hourly data with seasonal trends.
        """
        time = np.arange(hours)

        # 1. Wind Speed (Weibull distributed base + seasonal sin wave)
        # Winter higher, Summer lower
        seasonal_wind = 6.0 + 2.0 * np.sin(2 * np.pi * time / 8760 + np.pi / 2)
        noise_wind = np.random.weibull(2.0, hours) * 2.0
        wind_speed = np.clip(seasonal_wind + noise_wind - 2.0, 0, 25)

        # 2. Irradiance (Zero at night, peak at noon, seasonal intensity)
        # Seasonal peak intensity
        day_progress = (time % 24) / 24.0
        seasonal_solar = 0.8 + 0.2 * np.cos(2 * np.pi * time / 8760)  # Summer peak

        solar_profile = np.maximum(0, -np.cos(2 * np.pi * day_progress))  # Day/Night
        irradiance = 1000 * solar_profile * seasonal_solar * np.random.uniform(0.8, 1.0, hours)

        # 3. Temperature
        temp = 15 + 10 * np.sin(2 * np.pi * time / 8760 - np.pi / 2) + 5 * np.cos(2 * np.pi * day_progress)

        # 4. Electrical Load (Double peak profile typical of residential/grid)
        # Base load + Morning Peak + Evening Peak
        base_load = 2000  # kW
        daily_pattern = 1000 * (np.exp(-((time % 24 - 9) ** 2) / 10) + np.exp(-((time % 24 - 19) ** 2) / 10))
        load = base_load + daily_pattern + np.random.normal(0, 100, hours)

        df = pd.DataFrame({
            'wind_speed': wind_speed,
            'irradiance': irradiance,
            'temperature': temp,
            'load': load
        })

        return df


if __name__ == "__main__":
    loader = NASADataLoader()
    df = loader.generate_year_data()
    print(df.head())
    df.to_csv("synthetic_nasa_data.csv")