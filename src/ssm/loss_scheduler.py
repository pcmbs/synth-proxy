"""

For scheduling loss weights.

Adapted from https://github.com/hyakuchiki/SSSSM-DDSP/blob/main/diffsynth/schedules.py 

"""


class LinearScheduler:
    def __init__(self, start_value, end_value, start, warm):
        self.start_value = start_value
        self.end_value = end_value
        self.start = start
        self.warm = warm

    def __call__(self, i):
        l = max(i - self.start, 0)
        value = (self.end_value - self.start_value) * (float(l) / float(max(self.warm, l))) + self.start_value
        return value


class NopScheduler:
    def __init__(self, value):
        self.value = value

    def __call__(self, i):
        return self.value


class LossScheduler:
    def __init__(self, sched_cfg):
        self.schedules = {
            "param": self._get_schedule(sched_cfg, "param"),
            "perc": self._get_schedule(sched_cfg, "perc"),
        }

    @staticmethod
    def _get_schedule(sched_cfg, loss_name):
        loss_sched = sched_cfg.get(loss_name)
        if isinstance(loss_sched, float):
            return NopScheduler(loss_sched)
        if isinstance(loss_sched, dict):
            return LinearScheduler(**loss_sched)

        raise NotImplementedError()

    def get_schedules(self):
        return self.schedules["param"], self.schedules["perc"]
