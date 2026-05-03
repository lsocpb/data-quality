"""Pipeline service for keystroke analysis orchestration"""

import time
import sys
from pathlib import Path
import pandas as pd

# Add parent dir to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features import extract_per_key_features
from src.keystroke_knn import knn_predict, feature_columns
from src.keystroke_leave_one_out import evaluate_leave_one_out
from src.keystroke_identity import verify_claimed_identity, verification_score 
from .database_service import DatabaseService
from .models import AnalysisResult


class PipelineService:
    """Orchestrates keystroke analysis pipeline"""

    def __init__(self, database_service: DatabaseService):
        self.db = database_service
        self.training_df = None
        self.training_features = None
        self._initialize_training_data()

    def _initialize_training_data(self):
        """Load and extract features for all training data"""
        print("Loading training data...")
        raw_data = self.db.get_all_events_for_training()
        self.training_df = raw_data
        
        print(f"Extracting features from {len(raw_data)} events...")
        self.training_features = extract_per_key_features(raw_data)
        self.training_features.index.name = None  # Fix index name from pivot_table
        
        # Rename columns: 'a' -> 'hold_a', 'b' -> 'hold_b', etc.
        rename_cols = {}
        for col in self.training_features.columns:
            if col not in ['UserId', 'SampleNumber'] and len(col) == 1:
                rename_cols[col] = f'hold_{col}'
        
        if rename_cols:
            self.training_features = self.training_features.rename(columns=rename_cols)
        
        print(f"Training set: {len(self.training_features)} samples, {len(feature_columns(self.training_features))} features")

    def analyze_identification(
        self, 
        user_id: str, 
        sample_number: int, 
        metric: str = 'bray_curtis', 
        k: int = 5
    ) -> AnalysisResult:
        """
        Run full identification analysis: features + KNN + identification
        
        Args:
            user_id: User ID to analyze
            sample_number: Sample number for user
            metric: Distance metric ('euclidean', 'chebyshev', 'bray_curtis')
            k: Number of neighbors
            
        Returns:
            AnalysisResult with full pipeline output
        """
        timing = {}

        # Step 1: Get raw events
        t0 = time.time()
        events = self.db.get_sample_events(user_id, sample_number)
        timing['load_events'] = (time.time() - t0) * 1000

        if events.empty:
            return AnalysisResult(
                user_id=user_id,
                sample_number=sample_number,
                raw_event_count=0,
                features={},
                neighbors=[],
                predicted_user_id='ERROR',
                confidence_score=0.0,
                explanation='No events found for this sample',
                timing_ms=timing
            )

        # Step 2: Extract features for query sample
        t0 = time.time()
        sample_df = events.copy()
        sample_df['UserId'] = user_id
        sample_df['SampleNumber'] = sample_number
        
        query_features_df = extract_per_key_features(sample_df)
        
        if query_features_df.empty:
            return AnalysisResult(
                user_id=user_id,
                sample_number=sample_number,
                raw_event_count=len(events),
                features={},
                neighbors=[],
                predicted_user_id='ERROR',
                confidence_score=0.0,
                explanation='Could not extract features from events',
                timing_ms=timing
            )
        
        # Fix index name issue from pivot_table
        query_features_df.index.name = None
        
        # Rename columns to match training features
        rename_cols = {}
        for col in query_features_df.columns:
            if col not in ['UserId', 'SampleNumber'] and len(col) == 1:
                rename_cols[col] = f'hold_{col}'
        
        if rename_cols:
            query_features_df = query_features_df.rename(columns=rename_cols)
        
        # Convert to dict for output
        query_row = query_features_df.iloc[0]
        query_features_dict = {}
        for col in query_features_df.columns:
            if col.startswith('hold_'):
                key = col[5:]  # Remove 'hold_' prefix
                query_features_dict[key] = float(query_row[col])
        
        timing['extract_features'] = (time.time() - t0) * 1000

        # Step 3: KNN analysis
        t0 = time.time()
        
        try:
            # Get feature columns from training data
            train_feature_cols = feature_columns(self.training_features)
            
            # Ensure query_row has all feature columns (fill missing with 0)
            for col in train_feature_cols:
                if col not in query_row.index:
                    query_row[col] = 0.0
            
            # Get query vector
            query_vector = [float(query_row[col]) for col in train_feature_cols]
            
            # Run KNN prediction
            prediction = knn_predict(
                self.training_features,
                query_vector,
                k=k,
                metric=metric,
                user_col='UserId',
                sample_col='SampleNumber',
                feature_cols=train_feature_cols
            )
            
            neighbors = [(n['user'], n['distance'], n['sample']) for n in prediction.neighbors]
            predicted_user_id = str(prediction.predicted_user)
            confidence_score = float(prediction.score)
            
            # Build explanation
            if neighbors:
                top_neighbor = neighbors[0]
                explanation = f"Top match: {top_neighbor[0]} (distance: {top_neighbor[1]:.4f})"
            else:
                explanation = "No neighbors found"
        
        except Exception as e:
            predicted_user_id = 'ERROR'
            confidence_score = 0.0
            neighbors = []
            explanation = f"KNN Error: {str(e)}"

        timing['knn_analysis'] = (time.time() - t0) * 1000

        return AnalysisResult(
            user_id=user_id,
            sample_number=sample_number,
            raw_event_count=len(events),
            features=query_features_dict,
            neighbors=neighbors,
            predicted_user_id=predicted_user_id,
            confidence_score=confidence_score,
            explanation=explanation,
            metric=metric,
            k=k,
            timing_ms=timing
        )

    def analyze_verification(
        self,
        user_id: str,
        sample_number: int,
        claimed_user_id: str,
        threshold: float = 0.6,
        metric: str = 'bray_curtis',
        k: int = 3
    ) -> 'VerificationResult':
        """
        Run verification analysis using keystroke_identity.py
        Reuses professional verification implementation
        
        Returns: VerificationResult with verification decision
        """
        from .models import VerificationResult
        
        # Step 1: Get raw events
        events = self.db.get_sample_events(user_id, sample_number)
        
        if events.empty:
            return VerificationResult(
                user_id=user_id,
                sample_number=sample_number,
                raw_event_count=0,
                features={},
                neighbors=[],
                predicted_user_id='ERROR',
                confidence_score=0.0,
                explanation='No events found for this sample',
                timing_ms={},
                verification_match=False,
                threshold=threshold,
                claimed_user_id=claimed_user_id
            )
        
        # Step 2: Extract features for query sample
        sample_df = events.copy()
        sample_df['UserId'] = user_id
        sample_df['SampleNumber'] = sample_number
        
        query_features_df = extract_per_key_features(sample_df)
        
        if query_features_df.empty:
            return VerificationResult(
                user_id=user_id,
                sample_number=sample_number,
                raw_event_count=len(events),
                features={},
                neighbors=[],
                predicted_user_id='ERROR',
                confidence_score=0.0,
                explanation='Sample contains no identifiable key presses (only Unidentified/Special keys)',
                timing_ms={},
                verification_match=False,
                threshold=threshold,
                claimed_user_id=claimed_user_id
            )
        
        # Fix column naming to match training data
        query_features_df.index.name = None
        rename_cols = {}
        for col in query_features_df.columns:
            if col not in ['UserId', 'SampleNumber'] and len(col) == 1:
                rename_cols[col] = f'hold_{col}'
        
        if rename_cols:
            query_features_df = query_features_df.rename(columns=rename_cols)
        
        query_row = query_features_df.iloc[0]
        
        # Ensure query_row has all feature columns from training data (fill missing with 0)
        train_feature_cols = feature_columns(self.training_features)
        for col in train_feature_cols:
            if col not in query_row.index:
                query_row[col] = 0.0
        
        # Convert to dict for output
        query_features_dict = {}
        for col in query_features_df.columns:
            if col.startswith('hold_'):
                key = col[5:]  # Remove 'hold_' prefix
                query_features_dict[key] = float(query_row[col])
        
        # Step 3: Use professional verification from keystroke_identity.py
        try:
            verification_result = verify_claimed_identity(
                self.training_features,
                query_row,
                claimed_user=claimed_user_id,
                k=k,
                metric=metric,
                threshold=threshold,
                exclude_same_sample=False,
                user_col='UserId',
                sample_col='SampleNumber'
            )
            
            # Extract neighbors and predicted user for result
            neighbors = [(n['user'], n['distance'], n['sample']) for n in verification_result.prediction.neighbors]
            predicted_user_id = str(verification_result.predicted_user)
            confidence_score = float(verification_result.score)
            verification_match = verification_result.matched
            
            if neighbors:
                top_neighbor = neighbors[0]
                explanation = f"Verification {'PASSED' if verification_match else 'REJECTED'} - Score: {confidence_score:.4f} vs threshold: {threshold}"
            else:
                explanation = "No neighbors found"
        
        except Exception as e:
            predicted_user_id = 'ERROR'
            confidence_score = 0.0
            neighbors = []
            verification_match = False
            explanation = f"Verification Error: {str(e)}"
        
        return VerificationResult(
            user_id=user_id,
            sample_number=sample_number,
            raw_event_count=len(events),
            features=query_features_dict,
            neighbors=neighbors,
            predicted_user_id=predicted_user_id,
            confidence_score=confidence_score,
            explanation=explanation,
            timing_ms={},
            verification_match=verification_match,
            threshold=threshold,
            claimed_user_id=claimed_user_id
        )

    def leave_one_out_validation(
        self,
        k_values: list[int] = None,
        metrics: list[str] = None,
        progress_callback=None
    ) -> dict:
        """
        Leave-One-Out cross-validation using src/keystroke_leave_one_out.py
        Reuses professional sklearn-based implementation
        
        Note: Uses small k values (1,2,3) optimal for small datasets
        Larger k values (5+) require more training data than available
        
        Returns: {(k, metric): {accuracy, precision, recall, f1, tp, fp, fn}}
        """
        if k_values is None:
            k_values = [1, 2, 3]  # Optimized for small dataset (~4 samples/user)
        if metrics is None:
            metrics = ['euclidean', 'bray_curtis', 'chebyshev']
        
        # Filter training data (exclude empty users)
        training_data = self.training_features[self.training_features['UserId'] != ''].copy()
        
        try:
            # Use professional LOO implementation from src/keystroke_leave_one_out.py
            evaluation = evaluate_leave_one_out(
                training_data,
                ks=k_values,
                metrics=metrics,
                user_col='UserId',
                sample_col='SampleNumber'
            )
            
            # Convert summary_df to our format
            results = {}
            for _, row in evaluation.summary.iterrows():
                k = int(row['k'])
                metric = str(row['metric'])
                key = (k, metric)
                
                results[key] = {
                    'accuracy': float(row['accuracy']),
                    'precision': float(row['precision_macro']),
                    'recall': float(row['recall_macro']),
                    'f1': float(row['f1_macro']),
                    'tp': int(row['correct_predictions']),
                    'fp': int(row['iterations'] - row['correct_predictions']),
                    'fn': int(row['iterations'] - row['correct_predictions']),
                }
            
            return results
            
        except Exception as e:
            print(f"LOO Validation Error: {e}")
            raise

