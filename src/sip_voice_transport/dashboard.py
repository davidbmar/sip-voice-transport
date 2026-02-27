"""Simple call dashboard UI — served as a single HTML page from FastAPI.

Provides:
  GET /           — Dashboard HTML page (auto-refreshes via JS polling)
  GET /api/calls  — JSON endpoint for active and recent calls

No external dependencies — just HTML, CSS, and vanilla JS.
"""

import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)

# In-memory call tracking (not persistent — resets on restart)
_active_calls: dict[str, "CallInfo"] = {}
_recent_calls: list["CallInfo"] = []
MAX_RECENT = 50


@dataclass
class CallInfo:
    stream_id: str
    provider: str
    caller_id: str
    did: str
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    chunks_received: int = 0
    chunks_sent: int = 0
    status: str = "active"


def track_call_start(stream_id: str, provider: str, caller_id: str, did: str) -> CallInfo:
    """Register a new active call."""
    info = CallInfo(
        stream_id=stream_id,
        provider=provider,
        caller_id=caller_id,
        did=did,
    )
    _active_calls[stream_id] = info
    logger.info("Dashboard: tracking call %s", stream_id)
    return info


def track_call_end(stream_id: str) -> None:
    """Move a call from active to recent."""
    if stream_id in _active_calls:
        info = _active_calls.pop(stream_id)
        info.ended_at = time.time()
        info.status = "completed"
        _recent_calls.insert(0, info)
        if len(_recent_calls) > MAX_RECENT:
            _recent_calls.pop()
        logger.info("Dashboard: call ended %s", stream_id)


def track_audio(stream_id: str, direction: str) -> None:
    """Increment chunk count for a call."""
    if stream_id in _active_calls:
        if direction == "received":
            _active_calls[stream_id].chunks_received += 1
        elif direction == "sent":
            _active_calls[stream_id].chunks_sent += 1


def register_dashboard(app: FastAPI, did_router) -> None:
    """Register dashboard routes on a FastAPI app."""

    @app.get("/api/calls")
    async def api_calls():
        active = [asdict(c) for c in _active_calls.values()]
        recent = [asdict(c) for c in _recent_calls[:20]]
        return JSONResponse({
            "active": active,
            "recent": recent,
            "stats": {
                "active_count": len(_active_calls),
                "total_completed": len(_recent_calls),
            },
        })

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        did_numbers = did_router.dids if hasattr(did_router, "dids") else []
        did_rows = ""
        for did_num in did_numbers:
            cfg = did_router.lookup(did_num)
            if not cfg:
                continue
            # Escape HTML entities to prevent XSS
            safe_num = _html_escape(did_num)
            safe_name = _html_escape(cfg.display_name)
            safe_user = _html_escape(cfg.user_id)
            safe_model = _html_escape(cfg.llm_model)
            did_rows += f"""
                <tr>
                    <td><code>{safe_num}</code></td>
                    <td>{safe_name}</td>
                    <td>{safe_user}</td>
                    <td><code>{safe_model}</code></td>
                </tr>"""

        return DASHBOARD_HTML.replace("{{DID_ROWS}}", did_rows or """
                <tr><td colspan="4" style="color: #888; text-align: center;">
                    No DIDs configured — add them to sip_config.yaml
                </td></tr>""")


def _html_escape(s: str) -> str:
    """Escape HTML special characters."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIP Voice Transport</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            background: #0a0a0a;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 24px;
        }
        .header {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 32px;
            padding-bottom: 16px;
            border-bottom: 1px solid #222;
        }
        .header h1 {
            font-size: 20px;
            font-weight: 600;
            color: #fff;
        }
        .status-dot {
            width: 10px; height: 10px;
            border-radius: 50%;
            background: #444;
            transition: background 0.3s;
        }
        .status-dot.ok { background: #22c55e; box-shadow: 0 0 8px #22c55e44; }
        .status-dot.error { background: #ef4444; box-shadow: 0 0 8px #ef444444; }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }
        @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
        .card {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 20px;
        }
        .card h2 {
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 16px;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #1a1a1a;
        }
        .stat-row:last-child { border-bottom: none; }
        .stat-value {
            font-weight: 600;
            color: #fff;
            font-variant-numeric: tabular-nums;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th {
            text-align: left;
            padding: 8px 12px;
            color: #888;
            font-weight: 500;
            border-bottom: 1px solid #222;
        }
        td {
            padding: 8px 12px;
            border-bottom: 1px solid #1a1a1a;
        }
        code {
            background: #1a1a1a;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
        }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-active { background: #22c55e22; color: #22c55e; }
        .badge-completed { background: #3b82f622; color: #3b82f6; }
        .badge-telnyx { background: #8b5cf622; color: #8b5cf6; }
        .badge-twilio { background: #f5920022; color: #f59e0b; }
        .empty-state {
            color: #555;
            text-align: center;
            padding: 24px;
            font-style: italic;
        }
        .footer {
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid #222;
            color: #555;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
        }
        #last-update { color: #555; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <div class="status-dot" id="health-dot"></div>
        <h1>SIP Voice Transport</h1>
        <span id="last-update"></span>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Server Status</h2>
            <div class="stat-row">
                <span>Health</span>
                <span class="stat-value" id="health-status">...</span>
            </div>
            <div class="stat-row">
                <span>Ollama</span>
                <span class="stat-value" id="ollama-status">...</span>
            </div>
            <div class="stat-row">
                <span>Active Calls</span>
                <span class="stat-value" id="active-count">0</span>
            </div>
            <div class="stat-row">
                <span>Total Completed</span>
                <span class="stat-value" id="total-completed">0</span>
            </div>
        </div>

        <div class="card">
            <h2>Configured DIDs</h2>
            <table>
                <thead>
                    <tr><th>Number</th><th>Name</th><th>User</th><th>Model</th></tr>
                </thead>
                <tbody>
                    {{DID_ROWS}}
                </tbody>
            </table>
        </div>
    </div>

    <div class="card" style="margin-bottom: 24px;">
        <h2>Active Calls</h2>
        <div id="active-calls">
            <div class="empty-state">No active calls</div>
        </div>
    </div>

    <div class="card">
        <h2>Recent Calls</h2>
        <div id="recent-calls">
            <div class="empty-state">No recent calls</div>
        </div>
    </div>

    <div class="footer">
        <span>sip-voice-transport v0.1.0</span>
        <span>Polling every 2s</span>
    </div>

    <script>
        // Safe DOM helpers — no innerHTML, no XSS risk
        function el(tag, attrs, children) {
            const e = document.createElement(tag);
            if (attrs) Object.entries(attrs).forEach(([k, v]) => {
                if (k === 'className') e.className = v;
                else if (k === 'textContent') e.textContent = v;
                else e.setAttribute(k, v);
            });
            if (children) children.forEach(c => {
                if (typeof c === 'string') e.appendChild(document.createTextNode(c));
                else if (c) e.appendChild(c);
            });
            return e;
        }

        function formatDuration(start, end) {
            const seconds = Math.floor(((end || Date.now()/1000) - start));
            const m = Math.floor(seconds / 60);
            const s = seconds % 60;
            return m > 0 ? m + 'm ' + s + 's' : s + 's';
        }

        function providerBadge(provider) {
            const cls = provider === 'telnyx' ? 'badge badge-telnyx' : 'badge badge-twilio';
            return el('span', {className: cls, textContent: provider});
        }

        function codeEl(text) {
            return el('code', {textContent: text});
        }

        function renderCallTable(calls, showEnded) {
            const container = document.createDocumentFragment();

            if (!calls.length) {
                container.appendChild(el('div', {
                    className: 'empty-state',
                    textContent: 'No ' + (showEnded ? 'recent' : 'active') + ' calls'
                }));
                return container;
            }

            const headers = ['Provider', 'Caller', 'DID', 'Duration', 'Chunks'];
            if (showEnded) headers.push('Status');

            const headerRow = el('tr', null, headers.map(h => el('th', {textContent: h})));
            const thead = el('thead', null, [headerRow]);

            const rows = calls.map(c => {
                const cells = [
                    el('td', null, [providerBadge(c.provider)]),
                    el('td', null, [codeEl(c.caller_id || 'unknown')]),
                    el('td', null, [codeEl(c.did || 'unknown')]),
                    el('td', {textContent: formatDuration(c.started_at, c.ended_at)}),
                    el('td', {textContent: c.chunks_received + ' / ' + c.chunks_sent}),
                ];
                if (showEnded) {
                    cells.push(el('td', null, [
                        el('span', {className: 'badge badge-completed', textContent: 'completed'})
                    ]));
                }
                return el('tr', null, cells);
            });

            const tbody = el('tbody', null, rows);
            const table = el('table', null, [thead, tbody]);
            container.appendChild(table);
            return container;
        }

        function replaceChildren(parent, newContent) {
            while (parent.firstChild) parent.removeChild(parent.firstChild);
            if (newContent instanceof DocumentFragment || newContent instanceof HTMLElement) {
                parent.appendChild(newContent);
            }
        }

        async function poll() {
            try {
                const healthRes = await fetch('/health');
                const health = await healthRes.json();
                document.getElementById('health-status').textContent = health.status || 'unknown';
                document.getElementById('ollama-status').textContent = health.ollama ? 'Running' : 'Not running';
                document.getElementById('health-dot').className = 'status-dot ' + (health.status === 'ok' ? 'ok' : 'error');

                const callsRes = await fetch('/api/calls');
                const data = await callsRes.json();
                document.getElementById('active-count').textContent = data.stats.active_count;
                document.getElementById('total-completed').textContent = data.stats.total_completed;

                replaceChildren(document.getElementById('active-calls'), renderCallTable(data.active, false));
                replaceChildren(document.getElementById('recent-calls'), renderCallTable(data.recent, true));

                document.getElementById('last-update').textContent = 'Updated ' + new Date().toLocaleTimeString();
            } catch (e) {
                document.getElementById('health-dot').className = 'status-dot error';
                document.getElementById('health-status').textContent = 'Disconnected';
            }
        }

        poll();
        setInterval(poll, 2000);
    </script>
</body>
</html>
"""
