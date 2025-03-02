#!/bin/bash

# Run in development mode
python app.py

# Uncomment the following line to run in production mode with Gunicorn
# gunicorn -w 4 -b 0.0.0.0:5001 app:app 