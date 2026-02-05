#!/bin/bash

# Auto-ML Application Startup Script
# This script automates the setup and startup of the Streamlit Auto-ML application

set -e  # Exit on error

echo "=========================================="
echo "  Auto-ML Application Startup Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed. Please install Python 3 first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 3 found${NC}"

# Check if Homebrew is installed (for macOS libomp)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v brew &> /dev/null; then
        echo -e "${YELLOW}Warning: Homebrew not found. XGBoost may require libomp.${NC}"
        echo -e "${YELLOW}Install Homebrew: https://brew.sh${NC}"
    else
        echo -e "${GREEN}✓ Homebrew found${NC}"
        
        # Check if libomp is installed
        if brew list libomp &> /dev/null; then
            echo -e "${GREEN}✓ libomp is installed${NC}"
        else
            echo -e "${YELLOW}Installing libomp (required for XGBoost)...${NC}"
            brew install libomp
        fi
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip --quiet

# Install dependencies
echo -e "${YELLOW}Installing dependencies from requirements.txt...${NC}"
pip install -r requirements.txt

# Fix XGBoost library path on macOS (if needed)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ -f "venv/lib/python3.12/site-packages/xgboost/lib/libxgboost.dylib" ]; then
        echo -e "${YELLOW}Fixing XGBoost library path...${NC}"
        if [ -f "/opt/homebrew/opt/libomp/lib/libomp.dylib" ]; then
            install_name_tool -change @rpath/libomp.dylib /opt/homebrew/opt/libomp/lib/libomp.dylib venv/lib/python3.12/site-packages/xgboost/lib/libxgboost.dylib 2>/dev/null || true
            echo -e "${GREEN}✓ XGBoost library path fixed${NC}"
        fi
    fi
fi

echo ""
echo -e "${GREEN}=========================================="
echo "  Setup Complete!"
echo "==========================================${NC}"
echo ""
echo -e "${YELLOW}Starting Streamlit application...${NC}"
echo -e "${YELLOW}The app will open in your browser automatically.${NC}"
echo ""

# Start the Streamlit app
streamlit run app.py
