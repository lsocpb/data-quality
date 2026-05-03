"""Database service for loading keystroke data"""

import pandas as pd
from sqlalchemy import create_engine, Engine


class DatabaseService:
    """Service for querying keystroke data from PostgreSQL"""

    def __init__(self, engine: Engine):
        self.engine = engine

    def load_keystrokes(self, user_id: str | None = None) -> pd.DataFrame:
        """
        Load keystroke events from database
        
        Args:
            user_id: Filter by user ID (optional)
            
        Returns:
            DataFrame with columns: UserId, SampleNumber, KeyPressed, PressTime, ReleaseTime
        """
        query = 'SELECT * FROM "Keystrokes"'
        df = pd.read_sql(query, self.engine)
        
        if user_id:
            df = df[df['UserId'] == user_id]
        
        return df

    def load_users(self) -> list[str]:
        """Get list of all unique users in database"""
        query = 'SELECT DISTINCT "UserId" FROM "Keystrokes" ORDER BY "UserId"'
        result = pd.read_sql(query, self.engine)
        return result['UserId'].tolist()

    def load_samples_for_user(self, user_id: str) -> list[int]:
        """
        Get list of sample numbers for a specific user
        
        Args:
            user_id: User identifier
            
        Returns:
            List of sample numbers (sorted)
        """
        query = (
            f'SELECT DISTINCT "SampleNumber" FROM "Keystrokes" '
            f'WHERE "UserId" = \'{user_id}\' '
            f'ORDER BY "SampleNumber"'
        )
        result = pd.read_sql(query, self.engine)
        return result['SampleNumber'].tolist()

    def get_sample_events(self, user_id: str, sample_number: int) -> pd.DataFrame:
        """
        Get raw keystroke events for one sample
        
        Args:
            user_id: User identifier
            sample_number: Sample number
            
        Returns:
            DataFrame with keystroke events ordered by press time
        """
        query = (
            f'SELECT * FROM "Keystrokes" '
            f'WHERE "UserId" = \'{user_id}\' AND "SampleNumber" = {sample_number} '
            f'ORDER BY "PressTime"'
        )
        return pd.read_sql(query, self.engine)

    def get_all_events_for_training(self) -> pd.DataFrame:
        """Load all keystroke events (for KNN training set)"""
        return self.load_keystrokes()
