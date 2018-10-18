from metamer_mismatch_body import compute_metamer_mismatch_body, compute_object_color_solid
import colour
import numpy as np
from scipy.spatial import ConvexHull
import plotly.offline as py
import plotly.graph_objs as go

def mesh_from_hull(hull):
	vertices = hull.points
	faces = hull.simplices
	mesh = \
		go.Mesh3d(
		   x=np.array(vertices)[:,0],
		   y=np.array(vertices)[:,1],
		   z=np.array(vertices)[:,2],
		   i=np.array(faces)[:,0],
		   j=np.array(faces)[:,1],
		   k=np.array(faces)[:,2],
		   opacity=0.5)
	return mesh

def plot_convex_hulls(hulls, filename):
	layout = \
		go.Layout(
	        scene = dict(
	            xaxis = dict(range = [0,1]),
	            yaxis = dict(range = [0,1]),
	            zaxis = dict(range = [0,1]),
	            aspectmode='cube'))

	meshes = [mesh_from_hull(h) for h in hulls]

	py.plot(go.Figure(data=meshes, layout=layout), filename=filename)

# 400nm to 700nm and 10nm increments
standard_visible_range = colour.SpectralShape(400, 700, 10)

# setup two observers for observer induced metamerism example
XYZ_Observer_functions = \
	colour.STANDARD_OBSERVERS_CMFS['CIE 1964 10 Degree Standard Observer'].\
		interpolate(standard_visible_range).values

Nikon_Observer_functions = \
	colour.characterisation.CAMERAS_RGB_SPECTRAL_SENSITIVITIES['Nikon 5100 (NPL)'].\
		interpolate(standard_visible_range).values

# setup two illuminants for illuminant induced metamerism example
D65_Illuminant = \
	colour.ILLUMINANTS_RELATIVE_SPDS["D65"].\
		interpolate(standard_visible_range).values

A_Illuminant = \
	colour.ILLUMINANTS_RELATIVE_SPDS["A"].\
		interpolate(standard_visible_range).values

# compute the nikon response to an illuminated gray surface
gray_surface_reflectance = [0.5] * 31
gray_color_response_Φ = np.dot(Nikon_Observer_functions.T, np.multiply(D65_Illuminant, gray_surface_reflectance))

# Compute the metamer mismatch body for the following scenario:
# A 50% gray is viewed by a human under illuminant A, what is the MMB relative to a Nikon camera
# viewing the scene under D65. This is observer and illuminant induced metamerism.
mmb_surface_points = \
	compute_metamer_mismatch_body(
		observer_color_signal_Φ=gray_color_response_Φ,
		observer_response_functions_Φ=Nikon_Observer_functions,
		observer_response_functions_Ψ=XYZ_Observer_functions,
		scene_illumination_Φ=D65_Illuminant,
		scene_illumination_Ψ=D65_Illuminant)

mmb_convex_hull = ConvexHull(mmb_surface_points)

# compute the object color solid for XYZ observer and D65 illuminant, so it can be plotted
# in comparison
ocs_surface_points = \
	compute_object_color_solid(
		observer_response_functions=XYZ_Observer_functions,
		scene_illumination=D65_Illuminant)

ocs_convex_hull = ConvexHull(ocs_surface_points)

plot_convex_hulls([ocs_convex_hull, mmb_convex_hull], "mmb_plot.html")
