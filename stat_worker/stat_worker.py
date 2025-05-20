from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio

async def on_fetch(request, env):
    import asgi
    try:
        logger.debug(f"Request: {request}")
        logger.debug(f"Request type: {type(request)}")
        logger.debug(f"Request URL: {request.url}")
        logger.debug(f"Request method: {request.method}")
        logger.debug(f"Request headers type: {type(request.headers)}")
        logger.debug(f"Request headers: {dict(request.headers) if hasattr(request.headers, '__iter__') else 'Cannot convert headers to dict'}")
        
        # Check if this is a WebSocket upgrade request
        upgrade_header = request.headers.get('Upgrade')
        if upgrade_header and upgrade_header.lower() == 'websocket':
            logger.debug("WebSocket upgrade request detected")
            
        return await asgi.fetch(app, request, env)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise

app = FastAPI(debug=True)

import logging
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)

# List to store active connections
connections = []

@app.get("/")
async def root():
    return {"message": "WebSocket Stats Worker"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.debug("WebSocket endpoint handler invoked")
    try:
        logger.debug("About to accept WebSocket connection")
        await websocket.accept()
        logger.debug("WebSocket connection accepted successfully")
        connections.append(websocket)
    except Exception as e:
        logger.error(f"Error accepting WebSocket connection: {str(e)}")
        return
    
    try:
        count = 0
        while True:
            # Send a message every second
            count += 1
            message = f"Example data point {count}"
            await websocket.send_text(message)
            
            # HTMX compatible format - can be used with hx-sse-swap
            htmx_message = f"<div id='stats-update-{count}'>{message}</div>"
            await websocket.send_text(htmx_message)
            
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        connections.remove(websocket)
    finally:
        if websocket in connections:
            connections.remove(websocket)

# Simple HTML example to test the websocket
@app.get("/test")
async def test_page():
    from fastapi.responses import HTMLResponse
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebSocket Test</title>
        <script src="https://unpkg.com/htmx.org@1.9.5"></script>
    </head>
    <body>
        <h1>WebSocket Stats Test</h1>
        <div id="messages"></div>
        
        <script>
            // Use wss:// protocol if the page is loaded over https://, otherwise use ws://
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);
            const messagesDiv = document.getElementById('messages');
            
            socket.onmessage = function(event) {
                // Add new message to the div
                const message = event.data;
                if (message.startsWith('<')) {
                    // This is an HTMX compatible message
                    messagesDiv.innerHTML += message;
                } else {
                    // Plain text message
                    const messageElement = document.createElement('div');
                    messageElement.textContent = message;
                    messagesDiv.appendChild(messageElement);
                }
            };
            
            socket.onclose = function(event) {
                const messageElement = document.createElement('div');
                messageElement.textContent = 'Connection closed';
                messagesDiv.appendChild(messageElement);
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)