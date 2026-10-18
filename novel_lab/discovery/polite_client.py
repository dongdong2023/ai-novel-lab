"""礼貌爬取客户端：Gemini 评审 P1-3 改进。

诊断：原 discovery 模块高频轮询榜单，直连 requests 无
robots.txt 检查、无缓存、无退避，存在反不正当竞争法
风险（对经营性网站的高频抓取可能被认定为不正当竞争）。

PoliteClient 强制：
  1. robots.txt 合规：每次抓取前检查目标路径是否被 Disallow，
     被禁则拒绝请求并记录。
  2. 24h 缓存：同一 URL 24 小时内命中缓存直接返回，不发请求。
  3. 域名级限流：同一域名两次请求间隔 ≥15 秒。
  4. 退避：401/403/429 按指数退避（15s→60s→240s），
     连续 3 次失败后该域名冷却 1 小时。

用法：
    client = PoliteClient(cache_dir=DATA_DIR / "http_cache")
    resp = client.get("https://fanqienovel.com/rank/...")
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

# 域名级最小请求间隔（秒）
MIN_DOMAIN_INTERVAL = 15.0
# 缓存有效期（秒）：24h
CACHE_TTL = 24 * 3600
# 退避序列（秒）
BACKOFF_SEQUENCE = (15.0, 60.0, 240.0)
# 连续失败阈值 → 域名冷却 1 小时
MAX_CONSECUTIVE_FAILURES = 3
COOLDOWN_SECONDS = 3600.0

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)

#: 明确允许的域名白名单（榜单页）。白名单外域名拒绝抓取。
ALLOWED_DOMAINS = frozenset(
    {
        "fanqienovel.com",
        "www.fanqienovel.com",
        "www.qidian.com",
        "book.qidian.com",
    }
)


class PoliteClientError(Exception):
    pass


class RobotsDisallowed(PoliteClientError):
    pass


class DomainNotAllowed(PoliteClientError):
    pass


class DomainCoolingDown(PoliteClientError):
    pass


class PoliteClient:
    def __init__(
        self,
        cache_dir: Path | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: int = 30,
    ) -> None:
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.timeout = timeout
        self._last_request: dict[str, float] = {}
        self._failures: dict[str, int] = {}
        self._cooldown_until: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser] = {}
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent}
        )

    # ---------- robots.txt ----------

    def _robots_parser(self, domain: str) -> RobotFileParser:
        if domain not in self._robots:
            rp = RobotFileParser()
            rp.set_url(f"https://{domain}/robots.txt")
            try:
                rp.read()
            except Exception:
                # robots.txt 不可达 → 按最严格处理：拒绝
                rp = None
            self._robots[domain] = rp
        return self._robots[domain]

    def _check_robots(self, url: str) -> None:
        domain = urlparse(url).netloc
        rp = self._robots_parser(domain)
        if rp is None:
            raise RobotsDisallowed(
                f"{domain} 的 robots.txt 不可达，按严格模式拒绝"
            )
        if not rp.can_fetch(self.user_agent, url):
            raise RobotsDisallowed(
                f"robots.txt 禁止抓取: {url}"
            )

    # ---------- 缓存 ----------

    def _cache_path(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(
            url.encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, url: str) -> str | None:
        path = self._cache_path(url)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(
                path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        age = (
            datetime.now(timezone.utc).timestamp()
            - data.get("fetched_at", 0)
        )
        if age > CACHE_TTL:
            return None
        return data.get("body")

    def _cache_put(self, url: str, body: str) -> None:
        path = self._cache_path(url)
        if path is None:
            return
        path.write_text(
            json.dumps(
                {
                    "url": url,
                    "fetched_at": datetime.now(
                        timezone.utc
                    ).timestamp(),
                    "body": body,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # ---------- 限流与退避 ----------

    def _check_domain(self, url: str) -> str:
        domain = urlparse(url).netloc
        if domain not in ALLOWED_DOMAINS:
            raise DomainNotAllowed(
                f"域名不在抓取白名单: {domain}"
            )
        now = time.monotonic()
        if now < self._cooldown_until.get(domain, 0):
            raise DomainCoolingDown(
                f"域名 {domain} 冷却中（连续失败超限）"
            )
        return domain

    def _throttle(self, domain: str) -> None:
        last = self._last_request.get(domain, 0)
        wait = MIN_DOMAIN_INTERVAL - (
            time.monotonic() - last
        )
        if wait > 0:
            time.sleep(wait)

    def _record_success(self, domain: str) -> None:
        self._last_request[domain] = time.monotonic()
        self._failures[domain] = 0

    def _record_failure(
        self, domain: str, status: int | None
    ) -> float:
        """返回本次应退避的秒数。"""
        n = self._failures.get(domain, 0)
        self._failures[domain] = n + 1
        self._last_request[domain] = time.monotonic()
        if self._failures[domain] >= MAX_CONSECUTIVE_FAILURES:
            self._cooldown_until[domain] = (
                time.monotonic() + COOLDOWN_SECONDS
            )
            return COOLDOWN_SECONDS
        if status in (401, 403, 429):
            idx = min(n, len(BACKOFF_SEQUENCE) - 1)
            return BACKOFF_SEQUENCE[idx]
        return 0.0

    # ---------- 主入口 ----------

    def get(self, url: str) -> str:
        """GET 一个 URL，返回响应文本。

        顺序：白名单→robots→缓存→限流→请求→退避处理。
        """
        domain = self._check_domain(url)
        self._check_robots(url)

        cached = self._cache_get(url)
        if cached is not None:
            return cached

        self._throttle(domain)
        try:
            resp = self.session.get(
                url, timeout=self.timeout
            )
        except requests.RequestException as exc:
            backoff = self._record_failure(domain, None)
            raise PoliteClientError(
                f"请求失败 {url}: {exc}"
            ) from exc

        if resp.status_code in (401, 403, 429):
            backoff = self._record_failure(
                domain, resp.status_code
            )
            time.sleep(min(backoff, 60.0))
            raise PoliteClientError(
                f"{resp.status_code} 限流/拒绝 {url}，"
                f"已退避 {backoff:.0f}s"
            )

        if not resp.ok:
            self._record_failure(
                domain, resp.status_code
            )
            raise PoliteClientError(
                f"HTTP {resp.status_code} {url}"
            )

        self._record_success(domain)
        body = resp.text
        self._cache_put(url, body)
        return body
