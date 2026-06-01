from __future__ import annotations


class DemoTrainingsResults:
    def __init__(self, runs, metrics_no_ts, metrics_l2_l3, metrics_l2, metrics_l3):
        self.runs = runs
        self.metrics_no_ts = metrics_no_ts
        self.metrics_l2_l3 = metrics_l2_l3
        self.metrics_l2 = metrics_l2
        self.metrics_l3 = metrics_l3

