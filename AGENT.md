# FoosStat Agent Instructions

## Commands
- **Run server**: `uv run server.py` (FastAPI server on port 8080)
- **Install deps**: `uv sync`
- **Run analytics**: `uv run foos.py` (analyze game data from data.txt)

## Architecture
- **Main Python app**: FastAPI server (`server.py`) with WebSocket support for real-time foosball scoring
- **Frontend**: HTMX-based HTML templates for responsive UI (`splash.html`, `foosball_game_template.html`)
- **Legacy F# project**: `FoosStat2/` contains F# implementation with same analytics
- **Mobile app**: `mobile/foosstat/` React Native/Expo mobile version

## Code Style
- **Python**: Follow FastAPI patterns, use type hints, snake_case naming
- **HTML**: Tailwind CSS classes, HTMX attributes for interactivity
- **Game state**: Global state pattern with WebSocket broadcasting
- **Data format**: CSV-like possession tracking (b2,b3,b5,r2,r3,r5,g_b,g_r states)
- **Error handling**: Try/catch for WebSocket operations, graceful disconnection
