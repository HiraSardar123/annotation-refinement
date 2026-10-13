"""Small local web server for the annotation tool (standard library only).

    GET  /                      the app
    GET  /static/<file>         app files
    GET  /api/tree              experiments and their images
    GET  /api/image?key=...     one image: instances, label maps, conflicts
    GET  /api/photo?key=...     the source frame
    POST /api/save              save the edited result (RGB mask + COCO JSON)
    POST /api/save_folder       mark every remaining image of a folder reviewed, as it is

It only listens on 127.0.0.1, so nothing is reachable from other machines.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .data import Catalog, build_image, save_folder, save_image

STATIC = Path(__file__).resolve().parent / "static"
log = logging.getLogger("annotator")


def make_handler(cat: Catalog):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter console
            log.debug(fmt, *args)

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    return self._file(STATIC / "index.html")
                if url.path.startswith("/static/"):
                    target = (STATIC / url.path[len("/static/"):]).resolve()
                    if STATIC not in target.parents:
                        return self._json({"error": "not found"}, 404)
                    return self._file(target)
                if url.path == "/api/tree":
                    return self._json({"tree": cat.tree(), "outRoot": str(cat.out_root),
                                       "workRoot": str(cat.work_root)})
                if url.path == "/api/image":
                    return self._json(build_image(cat, q["key"]))
                if url.path == "/api/photo":
                    return self._file(cat.entries[q["key"]].image_path)
                return self._json({"error": "not found"}, 404)
            except KeyError as exc:
                return self._json({"error": f"unknown {exc}"}, 404)
            except Exception as exc:
                log.error("GET %s failed:\n%s", self.path, traceback.format_exc())
                return self._json({"error": str(exc)}, 500)

        def do_POST(self):
            url = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if url.path == "/api/save":
                    result = save_image(cat, payload)
                    log.info("saved %s (%d instances) -- %s: %d of %d reviewed",
                             result["mask"], result["instances"], result["experiment"],
                             result["reviewed"], result["total"])
                    if result["published"]:
                        log.info("all images of %s reviewed -> copied to %s",
                                 result["experiment"], result["published"])
                    return self._json(result)
                if url.path == "/api/save_folder":
                    result = save_folder(cat, payload["split"], payload["experiment"])
                    log.info("saved all remaining images of %s: %d saved, %d of %d reviewed",
                             result["experiment"], result["saved"], result["reviewed"], result["total"])
                    if result["published"]:
                        log.info("all images of %s reviewed -> copied to %s",
                                 result["experiment"], result["published"])
                    return self._json(result)
                return self._json({"error": "not found"}, 404)
            except Exception as exc:
                log.error("POST %s failed:\n%s", self.path, traceback.format_exc())
                return self._json({"error": str(exc)}, 500)

        def _file(self, path: Path) -> None:
            if not path.is_file():
                return self._json({"error": "not found"}, 404)
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send(200, path.read_bytes(), ctype)

    return Handler


class ExclusiveServer(ThreadingHTTPServer):
    """Refuses to share its port. The standard server sets SO_REUSEADDR, which
    on Windows lets a second copy bind the same port and silently take some of
    the requests (e.g. an old copy answering the new page with old data)."""
    allow_reuse_address = False

    def server_bind(self):
        import socket
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(cat: Catalog, port: int) -> ThreadingHTTPServer:
    return ExclusiveServer(("127.0.0.1", port), make_handler(cat))
