import os
import urllib.parse
import pandas as pd
from sqlalchemy import create_engine

# Database Details from your SSMS setup
SERVER_NAME = r"AMOGH-2000"
DATABASE_NAME = "HeartDiseaseDB"
TABLE_NAME = "heart_disease_risk"
OUTPUT_PATH = "data/processed/patient_records_v1.parquet"


def extract_snapshot():
    print(f"1. Connecting to SQL Server '{SERVER_NAME}', Database '{DATABASE_NAME}'...")

    # Build ODBC parameters safely using urllib
    params = urllib.parse.quote_plus(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={SERVER_NAME};"
        f"DATABASE={DATABASE_NAME};"
        f"Trusted_Connection=yes;"
    )

    # Windows Authentication Connection String
    connection_string = f"mssql+pyodbc:///?odbc_connect={params}"
    engine = create_engine(connection_string)

    print(f"2. Pulling data snapshot from dbo.{TABLE_NAME}...")
    query = f"SELECT * FROM dbo.{TABLE_NAME};"
    df = pd.read_sql(query, engine)

    print(f"   --> Extracted {len(df)} rows and {len(df.columns)} columns.")

    # Save to efficient Parquet format
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Data snapshot successfully saved to '{OUTPUT_PATH}'")


if __name__ == "__main__":
    extract_snapshot()