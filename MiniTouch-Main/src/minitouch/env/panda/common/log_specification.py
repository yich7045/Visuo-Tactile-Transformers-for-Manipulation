
class LogSpecification(object):
    def __init__(self, key, operation, frequency, log_name, operator_args=None):
        self.key = key
        self.operation = operation
        self.frequency = frequency
        self.log_name = log_name
        self.operator_args = operator_args