pass

class GroundTruthPerturbationEvaluationObject():
    """
    This class records errors associated with perturbed ground-truth predictions.
    Errors are recorded in the following format:
    errors = {
        'object_1': [[[x,y,z,theta], error], [[x,y,z,theta], error], ...], 
        'object_2': [[[x,y,z,theta], error], [[x,y,z,theta], error], ...],
        ...
        }
    """
    def __init__(self,
                distortion,
                scene_id,
                view_id,
                object_id,
                detection_id,
                coarse_error=None, 
                refiner_error=None,
                ):
        self.distortion = distortion
        self.scene_id = scene_id
        self.view_id = view_id
        self.object_id = object_id
        self.detection_id = detection_id
        self.coarse_error = coarse_error
        self.refiner_error = refiner_error

    def __str__(self) -> str:
        text = f"Distortion = {self.distortion}\nScene_id, view_id, detection_id, object_id = {self.scene_id}, {self.view_id}, {self.detection_id}, {self.object_id}\nCoarse error value: {self.coarse_error}\nRefiner error value: {self.refiner_error}"
        return text

    def update_coarse_error(self, coarse_error):
        self.coarse_error = coarse_error
    
    def update_refiner_error(self, refiner_error):
        self.refiner_error = refiner_error
    
