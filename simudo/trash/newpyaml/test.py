
from .test__pyaml import OpticalModel_

class OpticalModel(OpticalModel_):
    pass

/CB: Namespace()...

/CB/mixed: Namespace()

/optics

/optics/fields/w=.../

/CB/mixed/j/@mount: "SolutionVariable(_)"
/CB/mixed/j/element/degree: 1
/CB/mixed/j/element/space: "CG"

/CB/mixed/j
/CB/mixed/j/@trial_test_function: True

/trial/test: a

/CB/mixed/v -> j/test_function

# mount.get_by_suffix("@trial_test_function")
# all_by_suffix("@trial_test_function", ["/CB", "/VB", "/optics"])

/solution/test
/solution/trial
/solution/trial_components



