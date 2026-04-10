import http.server
import socket
import threading
from typing import Annotated
from pydantic import Field
from ecoscope_workflows_core.decorators import task


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass


@task
def start_cors_server(
    directory: str,
    port: int = 8099,
) -> Annotated[str, Field(description="URL of the running CORS server")]:
    handler = lambda *args: CORSRequestHandler(*args, directory=directory)
    try:
        server = http.server.HTTPServer(('127.0.0.1', port), handler)
    except socket.error as e:
        raise RuntimeError(f"Failed to bind to port {port}: {e}") from e
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/"
    print(f"CORS server running at {url} serving {directory}")
    return url
