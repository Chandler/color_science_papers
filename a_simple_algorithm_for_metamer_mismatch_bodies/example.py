from metamere_mismatch_body import compute_metamere_mismatch_body
import colour
import numpy as np
from scipy.spatial import ConvexHull
import plotly.offline as py
import plotly.graph_objs as go

def plot_3d_mesh(vertices, faces, filename):
	mesh = \
		go.Mesh3d(
		   x=np.array(vertices)[:,0],
		   y=np.array(vertices)[:,1],
		   z=np.array(vertices)[:,2],
		   i=np.array(faces)[:,0],
		   j=np.array(faces)[:,1],
		   k=np.array(faces)[:,2],
		   opacity=1)

	layout = \
		go.Layout(
	        scene = dict(
	            xaxis = dict(range = [0,1]),
	            yaxis = dict(range = [0,1]),
	            zaxis = dict(range = [0,1]),
	            aspectmode='cube'))

	py.plot(go.Figure(data=[mesh], layout=layout), filename=filename)

# 400nm to 700nm and 10nm increments
standard_visible_range = colour.SpectralShape(400, 700, 10)

# setup two observers and one illuminant for observer induced metamerism example
XYZ_Observer_functions = \
	colour.STANDARD_OBSERVERS_CMFS['CIE 1964 10 Degree Standard Observer'].\
		interpolate(standard_visible_range).values

Nikon_Observer_functions = \
	colour.characterisation.CAMERAS_RGB_SPECTRAL_SENSITIVITIES['Nikon 5100 (NPL)'].\
		interpolate(standard_visible_range).values

D65_Illuminant = \
	colour.ILLUMINANTS_RELATIVE_SPDS["D65"].\
		interpolate(standard_visible_range).values

A_Illuminant = \
	colour.ILLUMINANTS_RELATIVE_SPDS["A"].\
		interpolate(standard_visible_range).values

# green channel response to the illuminant is the typical scale factor
# nikon_scale_factor = np.dot(Nikon_Observer_functions.T, D65_Illuminant)[1]
XYZ_scale_factor = np.dot(XYZ_Observer_functions.T, D65_Illuminant)[1]

# compute the nikon response to an illuminated gray surface
gray_surface_reflectance = [0.5] * 31
color_response = np.dot(XYZ_Observer_functions.T, np.multiply(A_Illuminant, gray_surface_reflectance))

# compute the metamere mismatch body for the following scenario
# A 50% gray is viewed by a human under A_illuminant, what is the MMB relative to a Nikon camera
# viewing the scene under D65. This is observer *and* illuminant induced metamerism, usually you
# vary one or the other.
mmb_extrema_points = \
	compute_metamere_mismatch_body(
		observer_color_signal_Φ=color_response,
		observer_response_functions_Φ=XYZ_Observer_functions,
		observer_response_functions_Ψ=Nikon_Observer_functions,
		scene_illumination_Φ=A_Illuminant,
		scene_illumination_Ψ=D65_Illuminant)

# scale the result so that the illuminant color is the upper bound
relative_mmb_extrema_points = [p/XYZ_scale_factor for p in mmb_extrema_points]

mmb_convex_hull = ConvexHull(relative_mmb_extrema_points)

plot_3d_mesh(mmb_convex_hull.points, mmb_convex_hull.simplices, "mmb_plot.html")
