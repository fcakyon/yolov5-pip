import os
import re

from yolov5.utils.general import check_requirements


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


def resolve_roboflow_model_format(task: str) -> str:
    task_format_mapping = {
        "detect": "yolov5",
        "segment": "yolov5",
        "classify": "folder"
    }
    return task_format_mapping.get(task)


def check_dataset_roboflow(data: str, roboflow_token: str, task: str, location: str) -> str:
    if roboflow_token is None:
        raise ValueError("roboflow_token not found ❌")

    check_requirements("roboflow>=0.2.27")
    from roboflow import Roboflow

    workspace_name, project_name, project_version = extract_roboflow_metadata(url=data)
    os.environ["DATASET_DIRECTORY"] = location
    rf = Roboflow(api_key=roboflow_token)
    project = rf.workspace(workspace_name).project(project_name)
    model_format = resolve_roboflow_model_format(task=task)
    dataset = project.version(int(project_version)).download(
        model_format=model_format,
        overwrite=False
    )
    if task == "classify":
        return dataset.location
    return f"{dataset.location}/data.yaml"
