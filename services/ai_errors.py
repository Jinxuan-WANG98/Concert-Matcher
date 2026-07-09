from __future__ import annotations

from urllib.error import HTTPError, URLError


def describe_ai_exception(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        if exc.code == 403:
            return (
                "供应商拒绝访问（403）：Key 已配置，但账号余额、推理权限或模型权限未满足。"
                "请在供应商控制台完成实名认证、确认可用余额，或切换到已开通的模型。"
            )
        reason = f" {exc.reason}" if getattr(exc, "reason", None) else ""
        return f"供应商请求失败：HTTP {exc.code}{reason}"
    if isinstance(exc, URLError):
        return f"供应商连接失败：{exc.reason}"
    return str(exc)
