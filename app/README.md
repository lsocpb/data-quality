# Keystroke Dynamics Tkinter Application

Interactive GUI application for keystroke dynamics analysis - identification and verification of users based on keystroke biometrics.

## Features

- **Live Analysis**: Real-time keystroke analysis with configurable k-nearest neighbors, distance metrics, and results display
- **Results History**: Track and review all analysis results with detailed statistics
- **Registration Demo**: View user registration samples and keystroke event data
- **Multi-Metric Support**: Euclidean, Bray-Curtis, Chebyshev distance metrics
- **Adjustable Parameters**: Configure k (number of neighbors) and distance metric per analysis

## Project Structure

```
app/
├── main.py                    # Application entry point
├── config.py                  # Database configuration (loads DATABASE_URL from .env)
├── services/
│   ├── database_service.py    # Database access layer for keystroke data
│   ├── pipeline_service.py    # Pipeline orchestration (analysis logic)
│   └── models.py              # Data models (AnalysisResult, VerificationResult)
└── ui/
    ├── main_window.py         # Main Tkinter window with tabs
    ├── live_analysis.py       # Live Analysis view
    ├── results.py             # Results history and details view
    └── registration_demo.py   # Registration demo view
```

## Prerequisites

1. **Database**: PostgreSQL with keystroke data populated (via `fe-keystrokes` frontend and `KeystrokeAPI`)
2. **Environment**: `.env` file with `DATABASE_URL` configured
3. **Dependencies**: All in `pyproject.toml` (managed by `uv`)

### Example .env

```
DATABASE_URL=postgresql://username:password@localhost:5432/keystrokes_db
```

## Running the Application

### Option 1: Direct Python (Recommended for Development)

```bash
cd data-quality
uv run python run_app.py
```

### Option 2: Via Python Module

```bash
cd data-quality
uv run python -m app.main
```

### Option 3: Docker (TODO - requires Dockerfile)

## Usage

### 1. Live Analysis Tab

1. Select a **User** from the dropdown
2. Select a **Sample** (automatically populated based on user)
3. Choose a **Metric** (Euclidean, Bray-Curtis, Chebyshev)
4. Set **k** (number of neighbors to compare against)
5. Click **Analyze**
6. Results appear in the panel below showing:
   - Raw event count
   - Features extracted
   - Predicted user
   - Confidence score
   - Top k neighbors with distances
   - Total execution time

### 2. Results Tab

- View history of all analyses (last 50 retained)
- Double-click any result to see detailed information
- Treeview columns show:
  - Query User
  - Metric Used
  - k Value
  - Predicted User
  - Confidence Score
  - Total Timing

### 3. Registration Demo Tab

1. Select a **User**
2. View their registration samples (typically 5 samples for new users)
3. See event count and keystroke composition for each sample

## Architecture

### Data Flow

```
Database (PostgreSQL)
    ↓
DatabaseService (loads keystrokes, users)
    ↓
PipelineService (orchestrates analysis)
    ├─ Extract features (from src/features.py)
    ├─ Run KNN (from src/keystroke_knn.py)
    └─ Return AnalysisResult
    ↓
LiveAnalysisView (displays results)
    ↓
ResultsView (stores & displays history)
```

### Key Components

#### DatabaseService
- `load_users()` - Get all unique user IDs
- `load_keystrokes()` - Load all keystroke samples
- `load_samples_for_user(user_id)` - Get sample numbers for a user
- `get_sample_events(user_id, sample_number)` - Get raw keystroke events
- `get_all_events_for_training()` - Load all events for KNN training

#### PipelineService
- `analyze_identification(user_id, sample_number, metric, k)` - Run identification analysis
- `analyze_verification(user_id, sample_number, claimed_user_id, threshold, metric, k)` - Run verification

#### Models
- `AnalysisResult` - Identification result with neighbors and confidence
- `VerificationResult` - Extends AnalysisResult with verification status

## Testing

### Integration Test

```bash
cd data-quality
uv run python tests/test_integration.py
```

Tests:
- Service initialization
- Data loading
- Live analysis with multiple metrics
- Results validation
- Registration demo data loading
- Verification mode

## Performance

- **Training Data Initialization**: ~1-2 seconds (loads 11,602 keystroke events and extracts features for 71 training samples)
- **Single Analysis**: ~100-150ms (extract features + KNN with k=5)
- **Memory**: ~200-300MB (training data cached in memory for speed)

## Known Limitations

1. **Training Data**: Cached in memory after first initialization (not updated during app lifecycle)
2. **Database Queries**: Synchronous (no async support)
3. **Real-Time Streaming**: Not supported (uses static samples from database)
4. **Verification Threshold**: Fixed per analysis (not learned from data)

## Future Enhancements

1. Incremental training data updates
2. Async database queries for better UI responsiveness
3. Real-time keystroke streaming from live typing
4. User enrollment UI (capture new samples)
5. Performance metrics dashboard (accuracy, precision, recall)

## Architecture Notes

- **Reuses existing pipeline logic**: `src/features.py`, `src/keystroke_knn.py`, distance metrics
- **No database changes**: Uses existing PostgreSQL schema from `KeystrokeAPI`
- **Stateless pipeline**: Each analysis is independent (no state carried between calls)
- **Thread-based analysis**: Long-running analyses run on background threads to keep UI responsive

## Troubleshooting

### Application Won't Start
- Check `.env` file exists and `DATABASE_URL` is valid
- Test database connection: `psql $DATABASE_URL`
- Check Python version: Requires 3.11+

### No Users in Dropdown
- Verify keystroke data is populated in database
- Check `keystroke_events` table: `SELECT DISTINCT "UserId" FROM keystroke_events LIMIT 10;`

### Analysis Returns ERROR
- Check raw keystroke events for the sample: Look in database
- Verify all feature columns are numeric
- Check KNN input vectors have correct length

### Results Tab Shows Nothing
- Run an analysis first (Live Analysis tab)
- Results are displayed after "Analyze" completes
- Check status bar for error messages

## Contributing

When modifying services or views:
1. Update data models in `models.py` if result structure changes
2. Add tests in `tests/test_integration.py`
3. Update this README with new features
4. Follow existing code style (DRY, YAGNI, simple)

## References

- **Keystroke Pipeline**: `data-quality/README.md` (main pipeline documentation)
- **Feature Extraction**: `src/features.py`
- **KNN Implementation**: `src/keystroke_knn.py`
- **Data API**: `keystrokes/README.md`
- **Frontend**: `fe-keystrokes/README.md`
