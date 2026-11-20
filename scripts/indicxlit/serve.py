"""Live IndicXlit back-transliteration for the demo: a small HTTP service (Phase 11).

Runs INSIDE the IndicXlit container (Python 3.8, docker/indicxlit/); see
app/README.md for the docker command. The model is loaded once at start-up
(about a minute), then each request takes well under a second. Only Python's
standard library is used for the server, so the image needs nothing new.

    GET  /health      -> {"ok": true, "langs": [...], "beam": 10}
    POST /candidates  {"lang": "hi", "words": ["kisan", ...]}
                      -> {"kisan": [[candidate, p_model, p_word, score], ...], ...}

Candidates are scored exactly like the frozen candidate files (xlit_scoring.py),
so a word gets the same candidates live as in the experiments. The server
handles one request at a time: the engine is not thread-safe, and a demo has
one user.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from xlit_scoring import ScoredXlit

MAX_WORDS = 64  # per request; a search query has far fewer


def make_handler(xlit: ScoredXlit, langs):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # method name fixed by http.server
            if self.path == "/health":
                self._send(200, {"ok": True, "langs": list(langs), "beam": xlit.beam})
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):
            if self.path != "/candidates":
                self._send(404, {"error": "unknown path"})
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                lang, words = request["lang"], request["words"]
                if lang not in langs:
                    raise ValueError(f"language {lang!r} not loaded (loaded: {', '.join(langs)})")
                if not isinstance(words, list) or not all(isinstance(w, str) and w.isalpha() for w in words):
                    raise ValueError("words must be a list of alphabetic strings")
                if len(words) > MAX_WORDS:
                    raise ValueError(f"at most {MAX_WORDS} words per request")
            except (ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": str(e)})
                return
            words = sorted({w.lower() for w in words})
            scored, mismatches = xlit.candidates(words, lang) if words else ({}, 0)
            if mismatches:
                self._send(500, {"error": f"{mismatches} candidates ranked differently from IndicXlit"})
                return
            self._send(200, {w: [list(row) for row in rows] for w, rows in scored.items()})

        def log_message(self, fmt, *args):  # one short line per request
            print(f"{self.command} {self.path}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--langs", nargs="+", default=["hi", "gu"])
    parser.add_argument("--beam", type=int, default=10)   # as in the frozen candidate files
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"loading IndicXlit for {', '.join(args.langs)} ...", flush=True)
    xlit = ScoredXlit(args.langs, args.beam)
    server = HTTPServer(("0.0.0.0", args.port), make_handler(xlit, args.langs))
    print(f"ready on port {args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
