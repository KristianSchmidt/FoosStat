# Gemini Agent Instructions

This document provides instructions for the Gemini agent to interact with the FoosStat project.

## Commands

*   **Run server**: `uv run server.py`
*   **Install dependencies**: `uv sync`
*   **Run analytics**: `uv run foos.py`

## Architecture

*   **Main Python application**: A FastAPI server (`server.py`) that uses WebSockets for real-time foosball scoring.
*   **Frontend**: The frontend is built with HTMX-based HTML templates (`splash.html`, `foosball_game_template.html`).
*   **Legacy F# project**: The `FoosStat2/` directory contains a legacy F# implementation with the same analytics capabilities.

## Code Style

*   **Python**: Adhere to FastAPI best practices, use type hints, and follow the snake_case naming convention. Do not use shebangs in Python files.
*   **HTML**: Use Tailwind CSS classes and HTMX attributes for interactivity.
*   **Game state**: A global state pattern is used with WebSocket broadcasting.
*   **Data format**: Possession is tracked in a CSV-like format with the following states: `b2,b3,b5,r2,r3,r5,g_b,g_r`.
*   **Error handling**: Use try/catch blocks for WebSocket operations and handle disconnections gracefully.
