from deep_sort_realtime.deepsort_tracker import DeepSort
class FaceTracker():
    def __init__(self):
        self.tracker = DeepSort(max_age=10)