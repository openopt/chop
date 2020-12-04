from datetime import datetime


class Trace:
    """Trace callback"""

    def __init__(self, closure=None, log_x=True, log_grad=False, freq=1, callable=None):
        self.freq = int(freq)
        self.log_iterates = log_x
        self.closure = closure

        self.trace_x = []
        self.trace_time = []
        self.trace_step_size = []
        if log_grad:
            self.trace_grad = []
        if callable is not None:
            self.callable = callable
            self.trace_callable = []
        self.trace_f = []
        self.start = datetime.now()
        self._counter = 0

    def __call__(self, kwargs):
        if self.closure is None:
            self.closure = kwargs['closure']
        
        if self._counter % self.freq == 0:
            self.trace_x.append(kwargs['x'].data)
            self.trace_f.append(self.closure(kwargs['x'], return_jac=False).data)
            try:
                self.trace_callable.append(self.callable(kwargs))
            except AttributeError:
                pass

            try:
                self.trace_grad.append()
            except AttributeError:
                pass

            delta = (datetime.now() - self.start).total_seconds()
            self.trace_time.append(delta)
            self.trace_step_size.append(kwargs['step_size'])

        self._counter += 1
