from collections import defaultdict
import time
import itertools
from typing import Any, Optional
import rich.progress
import rich.table

def format_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds*1000000:.1f}μs"
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"

def metrics_style(m):
    if m == "loss":
        return "red"
    elif m == "accuracy":
        return "green"
    return ""

class ProgressBar:
    def __init__(self, prefix: str, final: float, metrics: list[tuple[str, str]]) -> None:
        self.prefix = prefix
        self.metrics = metrics
        self.start_time = None
        self.last_time = None
        self.final= final
        self.current = 0
        metric_columns = [ rich.progress.TextColumn("[" + metrics_style(metric) + "]{task.fields[m_" + metric + "]}", justify="center") for metric, label in metrics ]
        self._progress_bar = rich.progress.Progress(
            rich.progress.TextColumn("[bold blue]{task.description}"),
            rich.progress.BarColumn(bar_width=100),
            "[progress.percentage]{task.fields[percentage]}",
            rich.progress.TextColumn("{task.fields[n_out_of]}", justify="right"),
            "•",
            rich.progress.TextColumn("{task.fields[step_time]}", table_column=rich.table.Column(header="step time")),
            "•",
            rich.progress.TimeElapsedColumn(),
            *metric_columns,
        )
        self.last_metrics = {}

    def __enter__(self) -> None:
        self.start()
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_value is None:
            self.finish()
        else:
            self.finish(str(exc_value))

    def start(self) -> None:
        self.start_time = time.time()
        self.last_time = self.start_time
        self._progress_bar.start()
        fields: dict[str, Any] = {
            "step_time": "Time",
            "percentage": "",
            "n_out_of": ""
        }
        for metric, label in self.metrics:
            fields["m_" + metric] = label

        self.header_task = self._progress_bar.add_task("", start=False, total=1, **fields)
        empty: Any = ""
        self.task = self._progress_bar.add_task(self.prefix, start=True, total=self.final, **{ k:empty for k in fields.keys() })

    def finish(self, error: Optional[str] = None) -> None:
        assert isinstance(self._progress_bar.columns[1], rich.progress.BarColumn)
        assert self.start_time is not None
        # assert isinstance(self._progress_bar.columns[6], rich.progress.TimeRemainingColumn)

        if error:
            current = self.current
        else:
            self._progress_bar.columns[1].bar_width = 4
            self._progress_bar.update(self.task, percentage="100%")
            current = self.final
        # self._progress_bar.columns[6] = rich.progress.TimeElapsedColumn()
        self._progress_bar.update(self.header_task, completed=int(current == self.final and not error))
        self._progress_bar.update(self.task, completed=current, n_out_of=f"{current}/{self.final}", step_time=format_time((time.time() - self.start_time) / max(1, current)))
        self._progress_bar.stop()
        if error:
            self._progress_bar.console.print("[bold red]Error: " + error)

    def report_progress(self, completed: float, metrics: dict[str, float]) -> None:
        assert self.last_time is not None
        completed_num = completed - self.current
        if completed_num == 0:
            return
        self.current = completed
        step_time = time.time() - self.last_time
        unit_time = step_time / completed_num
        self.last_time = time.time()
        fields: dict[str, Any] = {
            "step_time": format_time(unit_time),
            "percentage": f"{(completed / self.final * 100):>3.1f}%",
            "n_out_of": f"{completed}/{self.final}"
        }
        self.last_metrics = metrics
        for metric, label in self.metrics:
            if metric in metrics:
                fields["m_" + metric] = f"{metrics[metric]:.3g}"
            else:
                fields["m_" + metric] = "-"
        self._progress_bar.update(self.task, completed=completed, **fields)


# p = ProgressBar("prefix", 1000, [("loss", "loss")])
# p.start()
# for i in range(100):
#     p.report_progress(i, {"loss": 1/(1+i)})
#     time.sleep(0.01)
# p.finish("neee")
