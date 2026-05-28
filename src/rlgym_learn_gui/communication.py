import sys
from typing import Any, Generic

import numpy as np
from requests import ConnectionError, get, post
from rlgym.api import ActionSpaceType, ObsSpaceType

URL = "http://localhost"


class GUICommunicator(Generic[ObsSpaceType, ActionSpaceType]):
    def __init__(self, port: int, name: str) -> None:
        self.name = name
        self.port = port

    def is_gui_alive(self) -> bool:
        try:
            _response = get(f"{URL}:{self.port}/")
        except ConnectionError:
            return False

        return _response.ok

    def send_metrics(self, project_id: str, run_name: str, metrics: dict[str, Any]):
        if not self.is_gui_alive():
            print("GUI couldn't be found, ignoring", file=sys.stderr)
            return

        for k in metrics.keys():
            if isinstance(metrics[k], np.floating):
                metrics[k] = float(metrics[k])

        post(
            f"{URL}:{self.port}/runs/{project_id}/{run_name}/metrics",
            json=metrics,
        )

    def set_spaces_types(
        self,
        session_id: str,
        agent_controller_name: str,
        obs_space: ObsSpaceType,
        act_space: ActionSpaceType,
    ):
        if not self.is_gui_alive():
            print("GUI couldn't be found, ignoring", file=sys.stderr)
            return

        post(
            f"{URL}:{self.port}/sessions/{session_id}/spaces",
            json={
                "agent_controller_name": agent_controller_name,
                "obs_space": obs_space,
                "act_space": act_space,
            },
        )
