import os
import re

from roboflow import Roboflow
from roboflow.core.version import Version
from typing import Dict, Optional

from yolov5.utils.plots import plot_results
from yolov5.utils.general import LOGGER

TASK2FORMAT: Dict[str, str] = {
    "detect": "yolov5",
    "segment": "yolov5",
    "classify": "folder"
}


def extract_roboflow_metadata(url: str) -> tuple:
    match = re.search(r'https://(?:app|universe)\.roboflow\.com/([^/]+)/([^/]+)(?:/dataset)?/([^/]+)', url)
    if match:
        workspace_name = match.group(1)
        project_name = match.group(2)
        project_version = match.group(3)
        return workspace_name, project_name, project_version
    else:
        raise ValueError(f"Invalid Roboflow dataset url ❌ "
                         f"Expected: https://universe.roboflow.com/workspace_name/project_name/project_version. "
                         f"Given: {url}")


class RoboflowConnector:

    project_version: Optional[Version] = None

    @staticmethod
    def init(url: str, roboflow_token: Optional[str]) -> None:
        if roboflow_token is None:
            raise ValueError("roboflow_token not found ❌")

        workspace_name, project_name, project_version = extract_roboflow_metadata(url=url)

        rf = Roboflow(api_key=roboflow_token)
        project_version = rf.workspace(workspace_name).project(project_name).version(int(project_version))
        RoboflowConnector.project_version = project_version

    @staticmethod
    def download_dataset(url: str, roboflow_token: Optional[str], task: str, location: Optional[str] = None) -> str:
        if roboflow_token is None:
            raise ValueError("roboflow_token not found ❌")

        if location:
            os.environ["DATASET_DIRECTORY"] = location
        RoboflowConnector.init(url=url, roboflow_token=roboflow_token)

        dataset = RoboflowConnector.project_version.download(
            model_format=TASK2FORMAT[task],
            overwrite=False
        )
        if task == "classify":
            return dataset.location
        return f"{dataset.location}/data.yaml"

    @staticmethod
    def upload_model(model_path: str):
        if RoboflowConnector.project_version is None:
            raise ValueError("RoboflowConnector must be initiated before you upload_model ❌")

        plot_results(file=os.path.join(model_path, "results.csv"))
        LOGGER.info(f"Uploading model from local: {model_path} to Roboflow: {RoboflowConnector.project_version.id}")
        RoboflowConnector.project_version.deploy(model_type="yolov5", model_path=model_path)
