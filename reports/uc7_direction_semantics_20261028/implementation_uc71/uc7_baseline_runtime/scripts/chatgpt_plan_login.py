#!/usr/bin/env python3
"""Foreground, loopback-only Sign in with ChatGPT; no inference or API billing.

On a remote workspace, securely forward local port 1455 to its loopback port
1455 before opening the printed /login URL. Never expose the helper publicly.
No OAuth callback, token, or authorization URL is printed or logged.
"""

from __future__ import annotations

import argparse
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt_field.chatgpt_auth import (AuthError, AuthorizationAttempt, DEFAULT_STORE,
                                  OpenAIAuthService, TokenStore)


def login(store: TokenStore, *, profile: str | None, new_account: bool,
          port: int, timeout: int) -> dict:
    if not 1 <= port <= 65535 or not 1 <= timeout <= 3600:
        raise AuthError("Use a valid loopback port and a timeout from 1 to 3600 seconds.")
    if profile and new_account:
        raise AuthError("Select a saved profile or a new account, not both.")
    status = store.status(profile)
    selected = None
    if profile or (status["connected"] and not new_account):
        selected = store.load(profile)
    host_id = store.host_id()
    service = OpenAIAuthService()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def page(self, status_code: int, message: str, *, button: bool = False):
            button_html = '<p><a href="/authorize">Continue with ChatGPT</a></p>' if button else ""
            body = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
                    "<title>GPT Field — ChatGPT sign-in</title><body>"
                    "<h1>GPT Field Robot Planning</h1><p>" + html.escape(message) + "</p>"
                    + button_html + "</body></html>").encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get("Host") != f"127.0.0.1:{port}":
                self.page(400, "Use the exact loopback sign-in URL printed by the helper.")
                return
            try:
                parsed = urlsplit(self.path)
                if parsed.path == "/login" and not parsed.query:
                    self.page(200, "Authorize this app to use your ChatGPT plan. Review its plan and credit limits in ChatGPT Settings before starting experiments.", button=True)
                elif parsed.path == "/authorize" and not parsed.query:
                    server.attempt = AuthorizationAttempt(host_id, f"http://127.0.0.1:{port}/auth/callback",
                                                           selected_record=selected,
                                                           registered_client_id=server.issued_client_id)
                    self.send_response(302)
                    self.send_header("Location", server.attempt.authorization_url())
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Referrer-Policy", "no-referrer")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif parsed.path == "/auth/callback":
                    if server.attempt is None:
                        raise AuthError("No browser authorization is pending. Return to /login.")
                    try:
                        record = server.attempt.accept_callback(parsed.query, service)
                        store.save(record, make_active=True)
                    finally:
                        # After a failed exchange, retry with the ID already issued
                        # to this attempt; never register another app accidentally.
                        if server.attempt.registered_client_id:
                            server.issued_client_id = server.attempt.registered_client_id
                    server.result = store.status(record["client_id"])
                    self.page(200, "ChatGPT plan authorization is saved securely. You may close this tab and return to the workspace.")
                else:
                    self.page(404, "Open /login to start ChatGPT authorization.")
            except AuthError as exc:
                self.page(400, str(exc) + " Return to /login to start a fresh attempt.")
                print("A browser sign-in attempt did not complete; retry at the local /login URL.", flush=True)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                # No tracebacks: HTTP request state can contain authorization codes.
                self.page(500, "The sign-in attempt failed. Return to /login and try again.")
                print("A browser sign-in attempt failed safely; no credentials were logged.", flush=True)

    try:
        server = HTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        raise AuthError("Cannot bind the requested loopback port. Select another port and forward that same port locally.") from None
    server.attempt = None
    server.issued_client_id = selected["client_id"] if selected else None
    server.result = None
    server.timeout = 1
    deadline = time.monotonic() + timeout
    print(f"Open http://127.0.0.1:{port}/login", flush=True)
    print("Remote workspace: forward this same port privately before opening the URL.", flush=True)
    print("Waiting for browser authorization. Tokens and callback URLs will not be printed.", flush=True)
    try:
        while server.result is None and time.monotonic() < deadline:
            server.handle_request()
        if server.result is None:
            raise AuthError("Browser authorization timed out; restart login to create a fresh attempt.")
        return server.result
    finally:
        server.server_close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE,
                        help="Private app credential directory; never a run/artifact directory")
    commands = parser.add_subparsers(dest="command", required=True)
    status_parser = commands.add_parser("status", help="Show safe local connection status; no network calls")
    status_parser.add_argument("--profile")
    login_parser = commands.add_parser("login", help="Run a temporary loopback-only OAuth listener")
    login_parser.add_argument("--profile", help="Reauthorize this existing issued client ID")
    login_parser.add_argument("--new-account", action="store_true")
    login_parser.add_argument("--port", type=int, default=1455)
    login_parser.add_argument("--timeout", type=int, default=600)
    select_parser = commands.add_parser("select", help="Make a saved account registration active")
    select_parser.add_argument("profile")
    args = parser.parse_args(argv)
    try:
        store = TokenStore(args.store_dir)
        if args.command == "login":
            result = login(store, profile=args.profile, new_account=args.new_account,
                           port=args.port, timeout=args.timeout)
        elif args.command == "select":
            store.select(args.profile)
            result = store.status()
        else:
            result = store.status(args.profile)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except AuthError as exc:
        print(f"ChatGPT sign-in: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("ChatGPT sign-in cancelled.", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
