from __future__ import annotations

from urllib.error import HTTPError, URLError


def describe_ai_exception(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        if exc.code == 403:
            return (
                "供应商拒绝访问（403）：Key 已配置，但该模型可能未开通、余额不足，或账号未完成实名认证。"
                "请确认控制台已开通当前模型；大模型（如 72B）常需单独权限，可改用 8B 等等级较低的视觉模型。"
            )
        reason = f" {exc.reason}" if getattr(exc, "reason", None) else ""
        return f"供应商请求失败：HTTP {exc.code}{reason}"
    if isinstance(exc, URLError):
        return f"供应商连接失败：{exc.reason}"
    return str(exc)
