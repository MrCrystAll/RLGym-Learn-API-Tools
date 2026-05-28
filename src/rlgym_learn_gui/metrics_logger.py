from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationInfo, model_validator
from rlgym_learn.api.typing import AgentControllerData
from rlgym_learn_algos.logging.dict_metrics_logger import DictMetricsLogger
from rlgym_learn_algos.logging.metrics_logger import (
    DerivedMetricsLoggerConfig,
    MetricsLogger,
)

from rlgym_learn_gui.communication import GUICommunicator

InnerMetricsLoggerConfig = TypeVar("InnerMetricsLoggerConfig")


def convert_nested_dict(d):
    new = {}
    for k, v in d.items():
        if isinstance(v, dict):
            converted = convert_nested_dict(v)
            to_add = {f"{k}/{k1}": v1 for k1, v1 in converted.items()}
        else:
            to_add = {k: v}
        new = {**new, **to_add}
    return new


class GUIMetricsLoggerConfig(BaseModel, Generic[InnerMetricsLoggerConfig]):
    run_name: str | None = None
    project_id: str | None = None
    port: int | None = None

    inner_metrics_logger_config: InnerMetricsLoggerConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_metrics_logger_config_model(cls, data: Any, info: ValidationInfo):
        inner_metrics_logger: MetricsLogger | None = info.context
        if inner_metrics_logger is None:
            return data

        if isinstance(data, dict) and "inner_metrics_logger_config" in data:
            inner_metrics_logger_config_raw = data["inner_metrics_logger_config"]

            if isinstance(inner_metrics_logger_config_raw, dict):
                metrics_logger_config_model_type: Type[Optional[BaseModel]] = (
                    inner_metrics_logger.config_model
                )
                if metrics_logger_config_model_type == type(None):
                    metrics_logger_config = None
                else:
                    metrics_logger_config = (
                        metrics_logger_config_model_type.model_validate(
                            inner_metrics_logger_config_raw,
                            context=inner_metrics_logger,
                        )
                    )
            else:
                metrics_logger_config = inner_metrics_logger_config_raw
            data["metrics_logger_config"] = metrics_logger_config

        return data


@dataclass
class GUIDerivedConfig:
    metrics_logger_config: GUIMetricsLoggerConfig


class GUIMetricsLogger(
    MetricsLogger[
        None,
        GUIMetricsLoggerConfig[InnerMetricsLoggerConfig],
        AgentControllerData,
    ],
    Generic[
        InnerMetricsLoggerConfig,
        AgentControllerData,
    ],
):
    def __init__(
        self,
        inner_metrics_logger: DictMetricsLogger[
            None, InnerMetricsLoggerConfig | None, AgentControllerData
        ],
        checkpoint_file_name: str = "gui_metrics_logger.json",
    ):
        self.inner_metrics_logger = inner_metrics_logger
        self.checkpoint_file_name = checkpoint_file_name

    @property
    def config_model(self) -> Type[GUIMetricsLoggerConfig]:
        """
        Function to return the config model type that your MetricsLogger implementation uses. Defaults to NoneType.
        """
        return GUIMetricsLoggerConfig

    def collect_env_metrics(self, data: List[Dict[str, Any]]):
        self.inner_metrics_logger.collect_env_metrics(data)

    def collect_agent_metrics(self, data: AgentControllerData):
        self.inner_metrics_logger.collect_agent_metrics(data)

    def report_metrics(self):
        _metrics = convert_nested_dict(self.inner_metrics_logger.get_metrics())
        self.gui_communicator.send_metrics(
            self.config.metrics_logger_config.project_id,
            self.config.metrics_logger_config.run_name,
            _metrics,
        )
        self.inner_metrics_logger.report_metrics()

    def load(self, config):
        self.config = config

        _run_name = config.metrics_logger_config.run_name
        if _run_name is None:
            raise ValueError(
                "No run name provided by the user nor by the agent controller"
            )

        _port = config.metrics_logger_config.port
        if _port is None:
            raise ValueError("No port provided by the user nor by the agent controller")

        self.gui_communicator = GUICommunicator(
            _port,
            "metrics_logger",
        )

        self.inner_metrics_logger.load(
            DerivedMetricsLoggerConfig(
                checkpoint_load_folder=config.checkpoint_load_folder,
                derived_agent_controller_config=config.derived_agent_controller_config,
                metrics_logger_config=config.metrics_logger_config.inner_metrics_logger_config,
            )
        )
