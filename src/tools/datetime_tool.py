"""
当前时间工具 / Current datetime tool.

支持任意 IANA 时区（如 Asia/Shanghai、America/New_York）。
Accepts any IANA timezone name (e.g. Asia/Shanghai, America/New_York).

术语 / Terminology:
    - IANA timezone: 标准时区数据库的字符串名（区分夏令时）
                     The standard timezone database identifier (DST-aware)
    - UTC: 协调世界时；不随夏令时变化，全球统一基准
            Coordinated Universal Time, the global reference timezone
    - zoneinfo: Python 3.9+ 标准库的时区模块（取代 pytz）
                Python 3.9+ stdlib timezone module (replaces pytz)
"""
from datetime import datetime, timezone

# ZoneInfo: IANA 时区对象 / IANA timezone object.
# ZoneInfoNotFoundError: 系统缺 tzdata 时抛 / Raised when tzdata missing.
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.tools.base import tool


@tool("Return current date and time. Accepts IANA timezone name (e.g. 'Asia/Shanghai'). Defaults to UTC.")
def get_current_datetime(tz_name: str = "UTC") -> str:
    try:
        # 默认或显式 UTC 走 datetime.timezone.utc（无依赖 tzdata）。
        # UTC fast path: use stdlib UTC singleton (no tzdata needed).
        tz = ZoneInfo(tz_name) if tz_name and tz_name.upper() != "UTC" else timezone.utc
    except ZoneInfoNotFoundError:
        # 系统缺 tzdata（多见于 Windows 极简环境）→ 回退到 UTC。
        # Fallback when tzdata missing (common on minimal Windows installs).
        tz = timezone.utc
        tz_name = "UTC (fallback – unknown timezone)"

    # datetime.now(tz) 返回 timezone-aware datetime（带时区信息）。
    # Returns a timezone-aware datetime.
    now = datetime.now(tz)
    # strftime 格式化字符串：
    #   %Y-%m-%d %H:%M:%S — 标准日期时间
    #   %Z                 — 时区缩写（"CST"、"PDT" 等）
    #   %A                 — 星期英文全名（"Monday"）
    # strftime format:
    #   %Y-%m-%d %H:%M:%S — date-time
    #   %Z                 — tz abbreviation
    #   %A                 — full weekday name in English
    return (
        f"Current datetime in {tz_name}: "
        f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')} "
        f"(weekday: {now.strftime('%A')})"
    )