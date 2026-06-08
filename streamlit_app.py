"""
Streamlit Cloud entry point.
This file must exist at the project root for Streamlit Cloud deployment.
It simply imports and runs the actual app from frontend/.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "frontend"))

from streamlit_app import main

if __name__ == "__main__":
    main()
