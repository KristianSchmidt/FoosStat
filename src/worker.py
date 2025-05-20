from workers import Response
import numpy as np

async def on_fetch(request, env):
    lst = np.random.randint(0, 100, size=100)
    return Response(f"Hello World!! {len(lst)}")