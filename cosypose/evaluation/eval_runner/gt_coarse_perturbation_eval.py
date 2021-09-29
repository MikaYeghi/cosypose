pass

class GroundTruthPerturbationEvaluation():
    """
    This class records errors associated with perturbed ground-truth predictions.
    Errors are recorded in the following format:
    errors = {
        'object_1': [[[x,y,z,theta], error], [[x,y,z,theta], error], ...], 
        'object_2': [[[x,y,z,theta], error], [[x,y,z,theta], error], ...],
        ...
        }
    """
    def __init__(self, coarse_errors={}, refiner_errors={}):
        self.coarse_errors = coarse_errors
        self.refiner_errors = refiner_errors
    
