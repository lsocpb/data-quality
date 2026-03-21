import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Engine

from src.config import DATABASE_URL

def get_db_engine() -> Engine:
    """SQLAlchemy engine to connect to the cloud database"""
    return create_engine(DATABASE_URL)

def fetch_keystrokes_data() -> pd.DataFrame:
    """Fetches the data from cloud database"""
    engine = get_db_engine()
    
    query = 'SELECT * FROM "Keystrokes" ORDER BY "Id" ASC'

    df = pd.read_sql(query, engine)

    return df


