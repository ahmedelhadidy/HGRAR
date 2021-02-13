class Error(Exception):
    pass


class ExceedingGRuleLength(Error):
    pass

class NotSupportedOperator(Error):
    pass