"""Pipeline package"""

try:
    from .realtime_prediction_service import RealtimePredictionService, get_prediction_service
    __all__ = ['RealtimePredictionService', 'get_prediction_service']
except ImportError:
    # Optional: realtime service requires additional dependencies (schedule, redis)
    __all__ = []
