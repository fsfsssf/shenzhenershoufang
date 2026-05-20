"""
异步任务管理器 — 管理找房任务的线程执行和人在回路通信
"""

import threading
import uuid
from typing import Dict, Optional, Callable

# 全局任务字典 — 存储所有异步任务状态
# 结构: {task_id: {"status": "...", "action": "...", "result": {...}, "error": "...", "confirmed": bool, "progress": "..."}}
_tasks: Dict[str, dict] = {}
_lock = threading.Lock()


def create_task() -> str:
    """
    创建一个新的异步任务
    
    生成一个8位UUID作为任务ID，初始化任务状态为 pending。
    
    Returns:
        task_id: 8位UUID字符串
    """
    task_id = str(uuid.uuid4())[:8]
    with _lock:
        _tasks[task_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "progress": "",
        }
    return task_id


def run_task(tid: str, func: Callable, *args, **kwargs):
    """
    在后台线程中执行任务
    
    将任务函数包装到后台线程中执行，避免阻塞主进程。
    任务执行完成后自动更新任务状态和结果。
    
    Args:
        tid   : 任务ID
        func  : 要执行的任务函数
        *args : 位置参数（传递给任务函数）
        **kwargs : 关键字参数（传递给任务函数）
    """
    def _run():
        """线程内部执行函数"""
        try:
            # 执行任务函数
            result = func(*args, **kwargs)
            # 更新任务状态为完成
            with _lock:
                if tid in _tasks:
                    _tasks[tid]["status"] = "done"
                    _tasks[tid]["result"] = result
        except Exception as e:
            # 更新任务状态为错误
            with _lock:
                if tid in _tasks:
                    _tasks[tid]["status"] = "error"
                    _tasks[tid]["error"] = str(e)

    # 先更新状态为 running
    with _lock:
        if tid in _tasks:
            _tasks[tid]["status"] = "running"

    # 创建并启动守护线程
    t = threading.Thread(target=_run, daemon=True)
    t.start()


def set_progress(task_id: str, message: str):
    with _lock:
        if task_id in _tasks:
            _tasks[task_id]["progress"] = message


def update_task_result(task_id: str, partial_result: dict):
    """
    实时更新任务的部分结果，用于增量展示
    
    Args:
        task_id: 任务ID
        partial_result: 部分结果字典，会与已有结果合并
    """
    with _lock:
        if task_id in _tasks:
            if _tasks[task_id]["result"] is None:
                _tasks[task_id]["result"] = {}
            _tasks[task_id]["result"].update(partial_result)


def get_task_status(task_id: str) -> Optional[dict]:
    with _lock:
        task = _tasks.get(task_id)
        if not task:
            return None
        return {
            "status": task["status"],
            "error": task.get("error"),
            "progress": task.get("progress", ""),
        }


def get_task_result(task_id: str):
    with _lock:
        task = _tasks.get(task_id)
        if not task:
            return None
        return task.get("result")
