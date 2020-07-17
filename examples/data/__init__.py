from pathlib import Path
import pandas as pd
import numpy as np


def get_stallion_data():
    def parse_yearmonth(df):
        return df.assign(date=lambda x: pd.to_datetime(x.yearmonth, format="%Y%m")).drop("yearmonth", axis=1)

    def clean_column_names(df):
        df.columns = [c.strip(" ").replace(" ", "_").replace("-", "_").lower() for c in df.columns]
        return df

    data_path = Path(__file__).parent.joinpath("stallion")
    weather = parse_yearmonth(clean_column_names(pd.read_csv(data_path / "weather.csv"))).set_index(["date", "agency"])
    price_sales_promotion = parse_yearmonth(
        clean_column_names(pd.read_csv(data_path / "price_sales_promotion.csv")).rename(
            columns={"sales": "price_actual", "price": "price_regular", "promotions": "discount"}
        )
    ).set_index(["date", "sku", "agency"])
    industry_volume = parse_yearmonth(clean_column_names(pd.read_csv(data_path / "industry_volume.csv"))).set_index(
        "date"
    )
    industry_soda_sales = parse_yearmonth(
        clean_column_names(pd.read_csv(data_path / "industry_soda_sales.csv"))
    ).set_index("date")
    historical_volume = parse_yearmonth(clean_column_names(pd.read_csv(data_path / "historical_volume.csv")))
    event_calendar = parse_yearmonth(clean_column_names(pd.read_csv(data_path / "event_calendar.csv"))).set_index(
        "date"
    )
    demographics = clean_column_names(pd.read_csv(data_path / "demographics.csv")).set_index("agency")

    # combine the data
    data = (
        historical_volume.join(industry_volume, on="date")
        .join(industry_soda_sales, on="date")
        .join(weather, on=["date", "agency"])
        .join(price_sales_promotion, on=["date", "sku", "agency"])
        .join(demographics, on="agency")
        .join(event_calendar, on="date")
        .sort_values("date")
    )
    for c in data.select_dtypes(object).columns:
        data[c] = data[c].astype("category")

    # minor feature engineering: add 12 month rolling mean volume
    data = data.assign(discount_in_percent=lambda x: (x.discount / x.price_regular).fillna(0) * 100)

    data["timeseries"] = data.groupby(["agency", "sku"], observed=True).ngroup()

    return data
