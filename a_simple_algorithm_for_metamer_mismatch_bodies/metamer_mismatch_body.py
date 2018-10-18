import numpy as np
from scipy import optimize
from numpy.testing import assert_array_almost_equal as almost_equal

COLOR_DIMENSIONS = 3
LIGHT_DIMENSIONS = 31

# vector representing a light beam with  power 1 at every wavelength
equal_energy_illumination_vector = [1] * LIGHT_DIMENSIONS

def assert_shape(m, shape):
    if m.shape != shape:
        raise ValueError("incorrect shape expected: {} found: {}".format(m.shape, shape))

def sample_unit_sphere(npoints):
    """
    return `npoints` random points on the unit sphere
    """
    vec = np.random.randn(3, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T

def solve_linear_program(
    object_function_coefficents,
    constraint_function=None,
    constraint_function_required_value=None,
    bounds=None):
    """
    This method minimizes and maximizes a linear function with respect to
    an equality constraint and lower and upper bounds

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
    Minimize: c^T * x
    Subject to:   
        A_ub * x <= b_ub
        A_eq * x == b_eq
    """
    xmax = \
        optimize.linprog(
            c=object_function_coefficents,
            A_eq=constraint_function,
            b_eq=constraint_function_required_value,
            bounds=bounds).x
    xmin = \
        optimize.linprog(
            c=object_function_coefficents * -1,
            A_eq=constraint_function,
            b_eq=constraint_function_required_value,
            bounds=bounds).x

    return (xmin, xmax)

def compute_metamer_mismatch_body(
    observer_color_signal_Φ,
    observer_response_functions_Φ,
    observer_response_functions_Ψ,
    scene_illumination_Φ=equal_energy_illumination_vector,
    scene_illumination_Ψ=equal_energy_illumination_vector,
    sampling_resolution=100):

    assert_shape(observer_color_signal_Φ,       (COLOR_DIMENSIONS,))
    assert_shape(observer_response_functions_Φ, (LIGHT_DIMENSIONS, COLOR_DIMENSIONS))
    assert_shape(observer_response_functions_Ψ, (LIGHT_DIMENSIONS, COLOR_DIMENSIONS))
    assert_shape(scene_illumination_Φ,          (LIGHT_DIMENSIONS,))
    assert_shape(scene_illumination_Ψ,          (LIGHT_DIMENSIONS,))

    color_signal_map_Φ = (observer_response_functions_Φ.T * scene_illumination_Φ).T
    color_signal_map_Ψ = (observer_response_functions_Ψ.T * scene_illumination_Ψ).T

    mismatch_body_extrema_points = []

    # iterate over a sampling of points on the unit sphere, interpreted as direction vectors
    # pointing in all directions.
    for direction_vector in sample_unit_sphere(sampling_resolution):

        # We assume the Euclidan Inner Product for our color vector space. Given that, a vector and its
        # functional (its covector) are identical. (This is why the euclidian dot product is a vector
        # matrix multiplied against itself)
        #
        # This functional can be thought of stacks of parallel
        # planes that are normal to `direction_vector`. Two of these planes will lie tangent to our metamer
        # mismatch body.
        direction_functional = direction_vector

        # compose the direction functional R3 -> R, with the color signal map to produce
        # a new funtional R31 -> R. 
        ΨF = np.dot(color_signal_map_Ψ, direction_functional)

        # construct a linear programming problem
        # equation to minimize and maximize: 
        #    ΨF, a function from R31 -> R
        #    ΨF returns the projection of some reflectance function in R31
        #    onto the line in R3 represented by `direction_vector`
        #
        # constraints:
        #    1) constrain the solution set of reflectances to  `0 > x_i <= 1`, this limits the solution to
        #       physically realizable reflectances
        #
        #    2) constrain the solution set to `color_signal_map_Φ(x) = observer_color_signal_Φ`, 
        #    this limits the solution to metamers of `observer_color_signal_Φ`
        #
        #    These are both convex sets. Their intersection is also a convex set, which is the metamer
        #    Mismatch Body we are computing.
        #
        min_reflectance, max_reflectance = \
            solve_linear_program(
                object_function_coefficents=ΨF,
                constraint_function=color_signal_map_Φ.T,
                constraint_function_required_value=observer_color_signal_Φ,
                bounds=(0,1))

        # # inline-test: these two reflectences should be metamers of `observer_color_signal_Φ`
        # almost_equal(observer_color_signal_Φ, np.dot(color_signal_map_Φ.T, min_reflectance), decimal=2)
        # almost_equal(observer_color_signal_Φ, np.dot(color_signal_map_Φ.T, max_reflectance), decimal=2)
        
        min_color_signal_Ψ = np.dot(color_signal_map_Ψ.T, min_reflectance) 
        max_color_signal_Ψ = np.dot(color_signal_map_Ψ.T, max_reflectance)

        mismatch_body_extrema_points.extend([min_color_signal_Ψ, max_color_signal_Ψ])

    # scale the resulting body so that the brightest illuminant color response == 1
    scale_factor = np.max(np.dot(observer_response_functions_Ψ.T, scene_illumination_Ψ))

    return [p/scale_factor for p in mismatch_body_extrema_points]

def compute_object_color_solid(
    observer_response_functions,
    scene_illumination=equal_energy_illumination_vector,
    sampling_resolution=100):
    """
    The linear programming formulation of the OCS is identical to that of the MMB minus
    the constraints related to the second observer.

    An MMB is a product of two observers but the OCS is simply the set of all object colors
    for a single observer.

    "Computing the object colour solid using spherical sampling"
    https://ueaeprints.uea.ac.uk/62975/
    """
    assert_shape(observer_response_functions, (LIGHT_DIMENSIONS, COLOR_DIMENSIONS))
    assert_shape(scene_illumination,          (LIGHT_DIMENSIONS,))

    color_signal_map = (observer_response_functions.T * scene_illumination).T

    ocs_extrema_points = []

    # iterate over a sampling of points on the unit sphere, interpreted as direction vectors
    # pointing in all directions.
    for direction_vector in sample_unit_sphere(sampling_resolution):

        direction_functional = direction_vector

        # compose the direction functional R3 -> R, with the color signal map to produce
        # a new funtional R31 -> R. 
        ΨF = np.dot(color_signal_map, direction_functional)

        min_reflectance, max_reflectance = \
            solve_linear_program(
                object_function_coefficents=ΨF,
                bounds=(0,1))

        min_color_signal = np.dot(color_signal_map.T, min_reflectance) 
        max_color_signal = np.dot(color_signal_map.T, max_reflectance)

        ocs_extrema_points.extend([min_color_signal, max_color_signal])

    # scale the resulting body so that the brightest illuminant color response == 1
    scale_factor = np.max(np.dot(observer_response_functions.T, scene_illumination))

    return [p/scale_factor for p in ocs_extrema_points]
