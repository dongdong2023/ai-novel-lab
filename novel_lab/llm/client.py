from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Callable

import requests


class LLMError(Exception):
    pass


class LLMDisabledError(LLMError):
    """llm.enabled=false 时的调用，P1 阶段这是预期行为。"""


class BudgetExceededError(LLMError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class LLMClient:
    """带回执与预算熔断的 LLM 客户端（借鉴 AIHOT receipts + budget）。

    - 回执复用：同 (purpose, prompt_hash, model) 的成功回执直接返回，
      重试不重复扣费。
    - 预算熔断：每分钟/小时/天请求上限，超限拒绝。
    - transport 可注入：测试用假服务，不访问外网。
    - P1 阶段 enabled=false，一切调用抛 LLMDisabledError，保证闭环零花费。
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        enabled: bool,
        base_url: str,
        api_key: str,
        model: str,
        per_minute_limit: int = 10,
        per_hour_limit: int = 200,
        per_day_limit: int = 2000,
        timeout: int = 600,
        extra_body: dict | None = None,
        transport: Callable[[str, str, str, str], str] | None = None,
    ):
        self.conn = conn
        self.enabled = enabled
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.limits = {
            "minute": per_minute_limit,
            "hour": per_hour_limit,
            "day": per_day_limit,
        }
        # 本地 27B 长推理需要大超时；extra_body 合并进请求体
        # （Ollama 的 think:false 关思考模式，提速数倍）
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self._transport = transport or self._http_transport

    def _http_transport(
        self, base_url: str, api_key: str, model: str, payload: str
    ) -> str:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text

    @staticmethod
    def _prompt_hash(system: str, user: str) -> str:
        digest = hashlib.sha256()
        digest.update(system.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(user.encode("utf-8"))
        return digest.hexdigest()

    def _check_budget(self) -> None:
        now = _now()
        windows = {
            "minute": now - timedelta(minutes=1),
            "hour": now - timedelta(hours=1),
            "day": now - timedelta(days=1),
        }
        for name, since in windows.items():
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM llm_receipts "
                "WHERE created_at >= ?",
                (since.isoformat(),),
            ).fetchone()
            if row["n"] >= self.limits[name]:
                raise BudgetExceededError(
                    f"LLM 预算熔断：{name} 窗口内已有 "
                    f"{row['n']} 次请求（上限 "
                    f"{self.limits[name]}）"
                )

    def _reuse_receipt(
        self, purpose: str, prompt_hash: str
    ) -> str | None:
        row = self.conn.execute(
            "SELECT response FROM llm_receipts "
            "WHERE purpose=? AND prompt_hash=? AND model=? "
            "AND status='ok' ORDER BY id DESC LIMIT 1",
            (purpose, prompt_hash, self.model),
        ).fetchone()
        return row["response"] if row else None

    def _write_receipt(
        self,
        purpose: str,
        prompt_hash: str,
        status: str,
        response: str,
    ) -> None:
        self.conn.execute(
            "INSERT INTO llm_receipts "
            "(purpose, model, prompt_hash, status, response,"
            " created_at) VALUES (?,?,?,?,?,?)",
            (
                purpose,
                self.model,
                prompt_hash,
                status,
                response,
                _now().isoformat(),
            ),
        )
        self.conn.commit()

    @staticmethod
    def _extract_content(raw: str) -> str:
        """从 OpenAI 兼容响应体里取 assistant 内容。

        取不到（非标准响应）就原样返回，由上层稳健解析兜底。
        """
        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"][
                "content"
            ]
            if isinstance(content, str):
                return content
        except (
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
        ):
            pass
        return raw

    def chat(
        self, system: str, user: str, *, purpose: str
    ) -> str:
        if not self.enabled:
            raise LLMDisabledError(
                "llm.enabled=false：模型调用被禁用，防止误扣费。"
                "P4 起需要真跑：在 novel_lab/config.toml 的 [llm] 里"
                "填 base_url/api_key/model 并把 enabled 改为 true；"
                "预算熔断上限也在同处配置。"
            )

        prompt_hash = self._prompt_hash(system, user)

        reused = self._reuse_receipt(purpose, prompt_hash)
        if reused is not None:
            return self._extract_content(reused)

        self._check_budget()

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **self.extra_body,
            },
            ensure_ascii=False,
        )

        try:
            raw = self._transport(
                self.base_url, self.api_key, self.model, payload
            )
        except Exception as exc:
            self._write_receipt(
                purpose, prompt_hash, "failed", str(exc)
            )
            raise LLMError(f"LLM 调用失败: {exc}") from exc

        self._write_receipt(
            purpose, prompt_hash, "ok", raw
        )

        return self._extract_content(raw)


def make_client(
    conn: sqlite3.Connection, config: dict
) -> LLMClient:
    cfg = config["llm"]
    return LLMClient(
        conn,
        enabled=cfg["enabled"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        per_minute_limit=cfg["per_minute_limit"],
        per_hour_limit=cfg["per_hour_limit"],
        per_day_limit=cfg["per_day_limit"],
        timeout=cfg.get("timeout_seconds", 600),
        extra_body=cfg.get("extra_body") or {},
    )
