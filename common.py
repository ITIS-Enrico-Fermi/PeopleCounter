
def context_error(f):
    """
    Decorator for error checking/handling.
    This way methods don't throw error messages,
    everything stays inside the context
    """
    def inner(*args, **kwargs):
        self = args[0]
        # Chain of waterfall methods is broken if error
        try:
            f(*args, **kwargs)
        except Exception as e:
            self._is_error = True
            self._error = e
        finally:
            return self
    return inner
