"""Agent-facing read tools for the Fluxer API (recall, search, members, audit).

Core's ``tools/discord_tool.py`` is hardwired to ``discord.com`` +
``DISCORD_BOT_TOKEN``, so the ``platform_toolsets.fluxer → hermes-discord``
mapping gives the agent tools aimed at the wrong API on a Fluxer deployment.
This module registers a plugin toolset ``fluxer`` whose tools hit
``FLUXER_BASE_URL`` with ``FLUXER_BOT_TOKEN`` instead. Read-only by design:
every mutation already has a first-class surface (reply, clarify cards).

Keep ``platform_toolsets.fluxer`` as ``[hermes-discord, fluxer]`` — the plugin
toolset alone is only the four ``flux_*`` tools and would lock the agent out of
shell/files. Handlers accept the gateway's single-positional-dict calling
convention via ``_merged``.
"""

from __future__ import annotations

import json
import os
import logging
import re
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("fluxer_platform.agent_tools")

_FLUXER_TOOLSET = "fluxer"
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def _get_secret(key: str, default: str = "") -> str:
    """Profile-scope secret read. Imported lazily: ``agent`` is a Hermes-core
    module that standalone CI does not install, and a missing scope must
    degrade to "unconfigured" instead of breaking collection/registration."""
    try:
        from agent.secret_scope import get_secret
    except ImportError:
        return default
    return (get_secret(key, default) or default)


def _api_base(base_url: str) -> str:
    """Normalize a Fluxer URL to the REST API base.

    Same rules as ``adapter._api_base`` — kept local so this module stays
    importable without the gateway stack that ``adapter`` pulls in.
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if (
        base.endswith("/api")
        or base.endswith("/api/v1")
        or base.endswith("/v1")
        or "/api/" in base
    ):
        return base
    if base == "https://api.fluxer.app":
        return f"{base}/v1"
    return f"{base}/api"


def _resolve() -> Tuple[str, str, str, str]:
    """(api_base, token, home_channel, guild_id) from the profile secret scope."""
    # Hosted default matches the adapter; already-scoped URLs are preserved.
    base = _api_base(_get_secret("FLUXER_BASE_URL")) or _api_base("https://api.fluxer.app")
    token = _get_secret("FLUXER_BOT_TOKEN").strip()
    home = _get_secret("FLUXER_HOME_CHANNEL").strip()
    guild = _get_secret("FLUXER_HOME_GUILD_ID").strip() or _derive_home_guild(base, token, home)
    return base, token, home, guild


def _fluxer_request(
    method: str, path: str, base: str, token: str,
    params: Optional[Dict[str, Any]] = None, body: Optional[Dict[str, Any]] = None,
    timeout: int = 15,
) -> Any:
    # ``base`` is already an API root (``…/v1`` or ``…/api``); do not append again.
    url = f"{base}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"refusing non-HTTP(S) Fluxer URL: {base!r}")
    req = urllib.request.Request(
        url, data=None if body is None else json.dumps(body).encode(), method=method,
        headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
    )
    # scheme is constrained to http(s) above; URL host comes from config, not the model
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        raw = resp.read(_MAX_RESPONSE_BYTES)
        return json.loads(raw.decode("utf-8")) if raw else None


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _merged(args: tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Gateway calls plugin handlers with the args dict as ONE positional;
    direct kwargs (unit tests / manual calls) merge over it."""
    first = args[0] if args else None
    merged: Dict[str, Any] = dict(first) if isinstance(first, dict) else {}
    merged.update(kwargs)
    return merged


_CHANNEL_GUILD_CACHE: Dict[str, Optional[str]] = {}


_SNOWFLAKE_RE = re.compile(r"\d{5,25}\Z")
_HOME_GUILD_CACHE: Dict[str, str] = {}


def _is_snowflake(value: str) -> bool:
    return bool(_SNOWFLAKE_RE.match(value))


def _cache_scope(base: str) -> str:
    """Isolate guild caches per API deployment and Hermes profile home.

    Channel snowflakes alone are not unique across restored backups or
    multiplexed profiles that share numeric ids against different servers.
    """
    profile_home = (os.environ.get("HERMES_HOME") or "").strip()
    return f"{base}|{profile_home}"


def _cache_key(base: str, channel_id: str) -> str:
    return f"{_cache_scope(base)}|{channel_id}"


def _derive_home_guild(base: str, token: str, home_channel: str) -> str:
    """Home guild id derived from the home channel when FLUXER_HOME_GUILD_ID
    is unset, so scoping fails closed even in minimal configs."""
    if not home_channel or not _is_snowflake(home_channel):
        return ""
    key = _cache_key(base, home_channel)
    cached = _HOME_GUILD_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        info = _fluxer_request("GET", f"/channels/{home_channel}", base, token)
    except Exception:
        return ""
    guild = str((info or {}).get("guild_id") or "")
    if guild:
        _HOME_GUILD_CACHE[key] = guild
    return guild


def _channel_allowed(channel_id: str, base: str, token: str, home_guild: str) -> bool:
    """Home-guild restriction for model-supplied channel ids.

    Default: a channel outside the home guild is refused, so a prompt-injected
    channel id cannot read out other channels the bot can see. Empty ids pass
    (callers substitute the home channel). ``FLUXER_TOOL_SCOPE=global`` widens.
    """
    if not channel_id:
        return True
    if _get_secret("FLUXER_TOOL_SCOPE").strip().lower() == "global":
        return True
    if not home_guild:
        return False  # no home guild derivable: refuse rather than read anywhere
    key = _cache_key(base, channel_id)
    if key in _CHANNEL_GUILD_CACHE:
        return _CHANNEL_GUILD_CACHE[key] == home_guild
    try:
        info = _fluxer_request("GET", f"/channels/{urllib.parse.quote(channel_id)}", base, token)
    except Exception:
        return False
    guild = str((info or {}).get("guild_id") or "")
    _CHANNEL_GUILD_CACHE[key] = guild or None
    return guild == home_guild


def _fmt_message(msg: Dict[str, Any]) -> str:
    author = (msg.get("author") or {}).get("username", "?")
    content = (msg.get("content") or "").replace("\n", " ")[:200]
    extras = []
    if msg.get("embeds"):
        extras.append(f"embeds={len(msg['embeds'])}")
    if msg.get("attachments"):
        extras.append(f"attachments={len(msg['attachments'])}")
    suffix = f" ({', '.join(extras)})" if extras else ""
    return f"{msg.get('id', '?')} [{msg.get('timestamp', '?')}] {author}: {content or '(no content)'}{suffix}"


def _flux_recall(*args: Any, **kwargs: Any) -> str:
    """Fetch the last messages from a Fluxer channel (defaults to the home channel)."""
    a = _merged(args, kwargs)
    channel_id = a.get("channel_id", "") or ""
    limit = a.get("limit", 20)
    before = a.get("before", "") or ""
    base, token, home, guild = _resolve()
    if not base or not token:
        return "Fluxer tools unavailable: FLUXER_BASE_URL / FLUXER_BOT_TOKEN not set."
    target = str(channel_id or home)
    if channel_id and not _is_snowflake(str(channel_id)):
        return "Fluxer recall failed: channel_id must be a channel id (digits)."
    if channel_id and not _channel_allowed(channel_id, base, token, guild):
        return "Fluxer recall refused: channel is outside the home guild (FLUXER_TOOL_SCOPE=global widens this)."
    params: Dict[str, Any] = {"limit": _clamp_int(limit, 20, 1, 100)}
    if before:
        params["before"] = str(before)
    try:
        msgs = _fluxer_request(
            "GET", f"/channels/{urllib.parse.quote(target)}/messages",
            base, token, params=params)
    except Exception as exc:
        return f"Fluxer recall failed: {exc}"
    if not isinstance(msgs, list):
        return "Fluxer recall failed: unexpected response shape."
    if not msgs:
        return "No messages in range."
    return "\n".join(_fmt_message(m) for m in msgs if isinstance(m, dict))


def _flux_search(*args: Any, **kwargs: Any) -> str:
    """Search messages in a Fluxer channel (defaults to the home channel)."""
    a = _merged(args, kwargs)
    query = a.get("query", "")
    channel_id = a.get("channel_id", "") or ""
    limit = a.get("limit", 10)
    base, token, home, guild = _resolve()
    if not base or not token:
        return "Fluxer tools unavailable: FLUXER_BASE_URL / FLUXER_BOT_TOKEN not set."
    if query is None or str(query).strip() == "":
        return "Fluxer search failed: query is required."
    target = str(channel_id or home)
    if channel_id and not _is_snowflake(str(channel_id)):
        return "Fluxer search failed: channel_id must be a channel id (digits)."
    if channel_id and not _channel_allowed(channel_id, base, token, guild):
        return "Fluxer search refused: channel is outside the home guild (FLUXER_TOOL_SCOPE=global widens this)."
    body: Dict[str, Any] = {
        "content": str(query),
        "context_channel_id": target,
        "hits_per_page": _clamp_int(limit, 10, 1, 50),
        "scope": "current",
    }
    try:
        data = _fluxer_request("POST", "/search/messages", base, token, body=body)
    except Exception as exc:
        return f"Fluxer search failed: {exc}"
    hits = []
    if isinstance(data, dict):
        hits = data.get("hits") or data.get("messages") or []
    if not hits:
        return "No hits."
    return "\n".join(_fmt_message(h) for h in hits if isinstance(h, dict))


def _flux_members(*args: Any, **kwargs: Any) -> str:
    """List members of the Fluxer home guild."""
    a = _merged(args, kwargs)
    limit = a.get("limit", 50)
    base, token, _, guild = _resolve()
    if not base or not token:
        return "Fluxer tools unavailable: FLUXER_BASE_URL / FLUXER_BOT_TOKEN not set."
    if not guild:
        return "Fluxer members unavailable: FLUXER_HOME_GUILD_ID not set."
    try:
        members = _fluxer_request(
            "GET", f"/guilds/{urllib.parse.quote(guild)}/members", base, token,
            params={"limit": _clamp_int(limit, 50, 1, 1000)})
    except Exception as exc:
        return f"Fluxer members failed: {exc}"
    if not isinstance(members, list):
        return "Fluxer members failed: unexpected response shape."
    if not members:
        return "No members returned."
    lines = []
    for m in members:
        if not isinstance(m, dict):
            continue
        user = m.get("user") or {}
        name = m.get("nick") or user.get("global_name") or user.get("username", "?")
        lines.append(f"{user.get('id', '?')} {name} (@{user.get('username', '?')})")
    return "\n".join(lines)


def _flux_audit(*args: Any, **kwargs: Any) -> str:
    """Recent audit-log entries for the Fluxer home guild (45-day retention)."""
    a = _merged(args, kwargs)
    limit = a.get("limit", 25)
    base, token, _, guild = _resolve()
    if not base or not token:
        return "Fluxer tools unavailable: FLUXER_BASE_URL / FLUXER_BOT_TOKEN not set."
    if not guild:
        return "Fluxer audit log unavailable: FLUXER_HOME_GUILD_ID not set."
    try:
        data = _fluxer_request(
            "GET", f"/guilds/{urllib.parse.quote(guild)}/audit-logs", base, token,
            params={"limit": _clamp_int(limit, 25, 1, 100)})
    except Exception as exc:
        return f"Fluxer audit log failed: {exc}"
    entries = data.get("audit_log_entries") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        return "No audit entries returned."
    users = {u.get("id"): u.get("username", "?")
             for u in (data.get("users") or []) if isinstance(u, dict)}
    lines = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        actor = users.get(e.get("user_id"), e.get("user_id", "?"))
        target = ""
        changes = e.get("changes") or []
        if changes and isinstance(changes[0], dict):
            target = str(changes[0].get("new_value", ""))
        lines.append(
            f"[{e.get('id', '?')}] action={e.get('action_type', '?')} actor={actor}"
            + (f" target={target}" if target else "")
            + (f" reason={e['reason']}" if e.get("reason") else ""))
    return "\n".join(lines)


_TOOLS = [
    {
        "name": "flux_recall",
        "description": "Read recent messages from a Fluxer channel (default: the home channel).",
        "schema": {
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Channel id (default home channel)"},
                "limit": {"type": "integer", "description": "1-100, default 20"},
                "before": {"type": "string", "description": "Only messages before this message id"},
            },
        },
        "handler": _flux_recall,
    },
    {
        "name": "flux_search",
        "description": "Full-text search of messages in a Fluxer channel (default: the home channel).",
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text"},
                "channel_id": {"type": "string", "description": "Channel id (default home channel)"},
                "limit": {"type": "integer", "description": "1-50 hits, default 10"},
            },
            "required": ["query"],
        },
        "handler": _flux_search,
    },
    {
        "name": "flux_members",
        "description": "List members of the Fluxer home guild (id, display name, username).",
        "schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "1-1000, default 50"},
            },
        },
        "handler": _flux_members,
    },
    {
        "name": "flux_audit",
        "description": "Recent audit-log entries for the Fluxer home guild (45-day retention).",
        "schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "1-100 entries, default 25"},
            },
        },
        "handler": _flux_audit,
    },
]


def register(ctx, *, register_fn: Optional[Callable] = None) -> int:
    """Register the ``fluxer`` toolset; returns the number of tools registered."""
    register_tool = register_fn or ctx.register_tool
    count = 0
    for tool in _TOOLS:
        try:
            register_tool(
                name=tool["name"],
                toolset=_FLUXER_TOOLSET,
                schema=tool["schema"],
                handler=tool["handler"],
                requires_env=["FLUXER_BOT_TOKEN"],
                description=tool["description"],
            )
            count += 1
        except Exception as exc:
            logger.warning("Fluxer tool %s not registered: %s", tool["name"], exc)
    return count
