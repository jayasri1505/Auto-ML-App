# Auto-ML Streamlit Application

An automated machine learning application built with Streamlit that allows users to upload datasets, perform exploratory data analysis, and automatically train multiple ML models with a downloadable pipeline.

## Features

- 📤 **Upload Dataset**: Upload CSV files for analysis
- 📊 **Exploratory Data Analysis**: Automated profiling with YData Profiling
- 🤖 **Automated ML**: Train multiple baseline models and compare accuracies
- 🔬 **TPOT AutoML**: Optional advanced AutoML with TPOT
- 💾 **Download Pipeline**: Download the best trained model pipeline

## Quick Start

### Prerequisites

- **Python 3.8+** installed
- **pip** available
- **macOS only**: Homebrew recommended (the startup script will install `libomp` for XGBoost)

### On macOS/Linux

From the project folder, run:

```bash
chmod +x start.sh
./start.sh
```

### On Windows

From the project folder, run:

```cmd
start.bat
```

### What the start script does

The script will automatically:
- ✅ Check for Python 3
- ✅ Create a virtual environment (if needed)
- ✅ Install all dependencies
- ✅ Fix XGBoost library paths (macOS)
- ✅ Start the Streamlit application

### Open the app

After starting, Streamlit will print a **Local URL** in the terminal (for example: `http://localhost:8501`). Open that URL in your browser.

## Manual Setup (Alternative)

If you prefer to set up manually:

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate virtual environment:**
   - macOS/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install libomp (macOS only, for XGBoost):**
   ```bash
   brew install libomp
   ```

5. **Fix XGBoost library path (macOS only):**
   ```bash
   install_name_tool -change @rpath/libomp.dylib /opt/homebrew/opt/libomp/lib/libomp.dylib venv/lib/python3.12/site-packages/xgboost/lib/libxgboost.dylib
   ```

6. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Project Documentation (Presentation Paper)

If you want a complete “read and understand everything” write-up (tech stack + modules + workflow),
see `PROJECT_OVERVIEW.txt`.

## Requirements

- Python 3.8 or higher
- pip
- (macOS) Homebrew (for libomp installation)

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── start.sh              # Startup script (macOS/Linux)
├── start.bat             # Startup script (Windows)
├── README.md             # This file
└── venv/                 # Virtual environment (created automatically)
```

## Usage

1. **Upload**: Upload your CSV dataset
2. **Profiling**: View automated exploratory data analysis
3. **ML**: Select target and features, then run modeling
4. **Download**: Download the best trained pipeline

## Troubleshooting

### XGBoost Error on macOS
If you see an XGBoost library error, make sure libomp is installed:
```bash
brew install libomp
```

### Port Already in Use
If port 8501 is busy, Streamlit will automatically use the next available port.

### Virtual Environment Issues
Delete the `venv` folder and run the startup script again to recreate it.

## License

This project is open source and available for educational purposes.
