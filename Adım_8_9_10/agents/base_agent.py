from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class AgentMessage:
    """
    Agent'lar arası iletişim birimi.
    task: ne yapılmasını isteniyor
    payload: girdi verisi
    result: çıktı verisi (execute sonrası dolar)
    error: hata varsa mesaj
    sender / receiver: agent kimliği
    """

    def __init__(
        self,
        task: str,
        payload: Dict[str, Any],
        sender: str = "unknown",
        receiver: str = "unknown",
    ) -> None:
        self.task = task
        self.payload = payload
        self.sender = sender
        self.receiver = receiver
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.success: bool = False

    def set_result(self, result: Dict[str, Any]) -> None:
        self.result = result
        self.success = True

    def set_error(self, error: str) -> None:
        self.error = error
        self.success = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "sender": self.sender,
            "receiver": self.receiver,
            "success": self.success,
            "result": self.result,
            "error": self.error,
        }


class BaseAgent(ABC):
    """
    Tüm agent'ların türediği soyut temel sınıf.

    Her agent:
    - Bir kimliğe (agent_id) sahiptir
    - execute() metodunu implement eder
    - AgentMessage alır, AgentMessage döndürür
    - Logging built-in
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"agent.{agent_id}")

    @abstractmethod
    def execute(self, message: AgentMessage) -> AgentMessage:
        """
        Gelen mesajı işler, sonucu mesaja yazar ve döndürür.

        Parameters
        ----------
        message : AgentMessage
            task ve payload dolu olarak gelir

        Returns
        -------
        AgentMessage
            result veya error set edilmiş halde
        """
        raise NotImplementedError

    def _success(self, message: AgentMessage, result: Dict[str, Any]) -> AgentMessage:
        """Başarılı sonucu mesaja yaz ve döndür."""
        message.set_result(result)
        self.logger.info("Task '%s' completed successfully.", message.task)
        return message

    def _failure(self, message: AgentMessage, error: str) -> AgentMessage:
        """Hata mesajını yaz ve döndür."""
        message.set_error(error)
        self.logger.error("Task '%s' failed: %s", message.task, error)
        return message
