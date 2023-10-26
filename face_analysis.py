from deepface import DeepFace
class FaceAnalysis():
    def __init__(self):
        self.deepface = DeepFace()
    def analyze(self,image):
        return self.deepface.analyze(img_path = image)
