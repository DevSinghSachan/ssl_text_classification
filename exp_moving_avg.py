

class ExponentialMovingAverage(object):
    """
    ExponentialMoving
    """
    def __init__(self, decay=0.999):
        self.decay = decay
        self.num_updates = 0
        self.shadow_variable_dict = {}

    def register(self, var_list):
        for name, param in var_list.items():
            self.shadow_variable_dict[name] = param.clone()

    def apply(self, var_list):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, param in var_list:
            if param.requires_grad:
                assert name in self.shadow_variable_dict
                data = self.shadow_variable_dict[name]
                data -= (1 - decay) * (data - param.data.clone())
