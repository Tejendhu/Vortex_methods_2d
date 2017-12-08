from __future__ import division
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.linalg
import csv
from copy import deepcopy
from numba import jit
from time import time
import sys
import csv
import math
@jit
def panel_midpoint(vertices):
	[vertex1, vertex2] = vertices
	return (vertex1 + vertex2)/2.0
@jit
def panel_length(vertices):
	[vertex1, vertex2] = vertices
	return abs(vertex2 - vertex1)
@jit
def panel_unit_vector(vertices):
	[vertex1, vertex2] = vertices
	return (vertex2 - vertex1)/panel_length(vertices)
@jit
def panel_unit_normal(vertices):
	return panel_unit_vector(vertices)*1j
@jit
def panel_velo(vertices, z, lamda0, lamda1):
	[vertex1, vertex2] = vertices
	unit_vector = panel_unit_vector(vertices)
	k = (vertex1 - z)/unit_vector
	vel = (lamda0 - (lamda1 * k)) * np.log((z - vertex2)/(z - vertex1))
	vel += lamda1 * panel_length(vertices) 
	vel *= 1j/(2*np.pi*unit_vector)
	vel = np.conj(vel)
	return vel
@jit
def panel_velg(vertices, z):
	[vertex1, vertex2] = vertices
	unit_vector = panel_unit_vector(vertices)
	length = panel_length(vertices)
	k = (vertex1 - z)/unit_vector
	multiplier = 1j/(2 * np.pi * unit_vector)
	c1 = length - (k * (np.log(1 + (length/k) )))
	c1 *= multiplier
	c0 = np.log((z - vertex2)/(z - vertex1))
	c0 *= multiplier
	return  np.conj(c0), np.conj(c1)

@jit
def vortices_velo_chorin(free_vortices_gamma, free_vortices_pos,free_vortices_core_radius, pos):
	if len(free_vortices_gamma) == 0:
		return 0. + 0j
	velocity = (-.5*1j/np.pi)*(free_vortices_gamma/(pos - free_vortices_pos))
	r = abs(pos - free_vortices_pos)
	velocity = np.conj(velocity) * ((r**2)/(r**2 + free_vortices_core_radius**2))
	return velocity.sum()

@jit
def rvm_fullsplit(vor1_gamma, vor1_pos, vor1_core_radius, delta_t, viscosity, gamma_max):
	gamma_max = float(gamma_max)
	gamma = float(vor1_gamma)
	no_of_splits = int(abs(gamma)/gamma_max) 
	if no_of_splits <= 1:
		no_of_splits = 1
	new_gamma = gamma/no_of_splits
	z0, core_radius = vor1_pos, vor1_core_radius
	new_x, new_y = np.random.randn(no_of_splits), np.random.randn(no_of_splits)
	new_z = new_x + 1j*new_y
	new_z *= 2*viscosity*delta_t
	new_vortices_pos = new_z + z0
	new_vortices_gamma = np.zeros(no_of_splits) + new_gamma
	new_vortices_core_radius = np.zeros(no_of_splits) + core_radius 
	return  new_vortices_gamma, new_vortices_pos, new_vortices_core_radius

def plot_vortex_positions(vortices_gamma, vortices_pos, name):
	color = ['r' if value > 0 else 'b' for value in vortices_gamma]
	plt.scatter(vortices_pos.real, vortices_pos.imag , color = color, s = 5)
	plt.xlim(-1.5, 6)
	plt.ylim(-3, 3.25)
	plt.gca().set_aspect('equal', adjustable='box')
	plt.savefig(name + ".png")	
	plt.close()
	return 

# @jit
def get_em_drunk(vortices_gamma, vortices_pos, vortices_core_radius, delta_t, viscosity, gamma_max):
	new_vortices_gamma,new_vortices_pos,new_vortices_core_radius = np.array([],dtype = float), np.array([],dtype = complex), np.array([], dtype = float)
	for i in range(len(vortices_gamma)):
		new = rvm_fullsplit(vortices_gamma[i], vortices_pos[i], vortices_core_radius[i], delta_t, viscosity, gamma_max)
		new_vortices_gamma = np.append(new_vortices_gamma, new[0])
		new_vortices_pos = np.append(new_vortices_pos, new[1])
		new_vortices_core_radius = np.append(new_vortices_core_radius, new[2])
	# new_vortices = np.array([new_vortices_gamma, new_vortices_pos, new_vortices_core_radius])
	# print new_vortices
	return new_vortices_gamma, new_vortices_pos, new_vortices_core_radius

def create_panels_for_circle(z0, r, no_of_panels):#working
	circle = []
	panels = []
	for i in range(no_of_panels):
		angle = -2*np.pi/no_of_panels  # minus is to ensure clockwise sense of panels
		vertex1 = z0 + (r * complex(np.cos(i*angle), 1.2*np.sin(i*angle)))
		vertex2 = z0 + (r * complex(np.cos((i+1)*angle), 1.2*np.sin((i+1)*angle)))	
		circle.append(vertex1)
		panels.append([vertex1, vertex2])
	circle.append(vertex2)
	circle = np.array(circle)
	panels = np.array(panels)
	return panels, circle

def create_panels_csvfile(filename, AoA = 0):#working
	aerofoil = []
	file = open(filename, 'rb')
	for a in file:
		b = a.rstrip().split(",")
		aerofoil.append(complex(float(b[0]), float(b[1])))
	aerofoil = np.array(aerofoil)
	AoA *= -1 * np.pi/180.0
	aerofoil = aerofoil * complex(np.cos(AoA), np.sin(AoA))
	panels = []
	for i in range(len(aerofoil)-1):
		p = panel(aerofoil[i], aerofoil[i+1])
		panels.append(aerofoil[i], aerofoil[i+1])
	return panels, aerofoil

def dot_product_complex(a, b):#working
	return (a.real * b.real) + (a.imag * b.imag)
# @jit
def get_N_n(all_panels):
	no_of_bodies = len(all_panels)
	N = 0
	n =[]
	for panels in all_panels:
		no_of_panels = len(panels)
		N += no_of_panels
		n.append(no_of_panels)
	return N, n
# @jit
def get_matrix_A(all_panels):
	N, n = get_N_n(all_panels)
	A = np.zeros((2*N+len(n), 2*N))
	body_number1 = 0
	panel_number1 = 0
	for i in range(N):
		if panel_number1 > n[body_number1] - 1:
			body_number1 += 1
			panel_number1 = 0
		A[i + N][i], A[i + N][i - panel_number1 + ((panel_number1 + 1)%n[body_number1])] = 1, -1
		length_panel = panel_length(all_panels[body_number1][panel_number1])
		A[i + N][i + N] = length_panel
		body_number2 = 0
		panel_number2 = 0
		for j in range(N):
			if panel_number2 > n[body_number2] - 1:
				body_number2 += 1
				panel_number2 = 0
			control_point = panel_midpoint(all_panels[body_number1][panel_number1])
			a = panel_velg(all_panels[body_number2][panel_number2], control_point)
			A[i][j] = dot_product_complex(a[0], panel_unit_normal(all_panels[body_number1][panel_number1]))
			A[i][j+N] = dot_product_complex(a[1], panel_unit_normal(all_panels[body_number1][panel_number1]))
			panel_number2 +=1
		for j in range(len(n)):
			if body_number1 == j:
				A[-1*len(n)+j][i], A[-1*len(n)][i+N] = length_panel, .5*(length_panel**2)
		panel_number1 += 1

	return A
# @jit
def get_matrix_B(all_panels, free_stream_velocity, free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, circulation, body_velocity = 0):
	N, n = get_N_n(all_panels)
	B = np.zeros(2*N + len(n))
	body_number = 0
	panel_number = 0
	for i in range(N):
		if panel_number > n[body_number] - 1:
			body_number += 1
			panel_number = 0
		control_point = panel_midpoint(all_panels[body_number][panel_number])
		b = body_velocity -1 * free_stream_velocity 
		b -= vortices_velo_chorin(free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, control_point)
		v = all_panels[body_number][panel_number]

		# print v
		
		a = panel_unit_normal(v)
		b = dot_product_complex(b, a)
		B[i] = b
		# print panel_number,body_number
		panel_number += 1
	for i in range(len(n)):
		B[i -len(n)] = circulation[i]

	return B

def solve_panels(all_panels, free_stream_velocity,free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, circulation, body_velocity = 0):
	B = get_matrix_B(all_panels, free_stream_velocity, free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, circulation, body_velocity)
	A = get_matrix_A(all_panels)
	X = np.array(scipy.linalg.lstsq(A,B)[0])
	return X
# @jit
def divide_X_panelwise(all_panels, X):
	N, n = get_N_n(all_panels)		
	body_counter, panel_counter = 0, 0
	Y = [[]]
	for i in range(N):
		if panel_counter > n[body_counter] - 1:
			panel_counter = 0
			body_counter += 1
			Y.append([])
		Y[-1].append([X[i], X[i+N]]) 
		panel_counter += 1
	return Y
#####################################################################################################
# @jit
def add_noslip_blobs123(all_panels, free_stream_velocity, free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, circulation, body_velocity ):
	X = solve_panels(all_panels, free_stream_velocity, free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, circulation, body_velocity)
	Y = divide_X_panelwise(all_panels, X)
	N, n = get_N_n(all_panels)
	noslip_blobs = []
	body_counter = 0
	f1,f2,f3 = free_vortices_gamma, free_vortices_pos, free_vortices_core_radius
	for body in all_panels:
		panel_counter, length = 0, 0

		for panel in body:
			length += abs(panel[0] - panel[1])
		length = length/len(body) 
		for panel_vertices in body:
			delta_n = length / np.pi
			noslip_blob_position = panel_midpoint(panel_vertices) + (delta_n * panel_unit_normal(panel_vertices))
			noslip_blob_gamma = Y[body_counter][panel_counter][0]*length + (Y[body_counter][panel_counter][1]*(length**2)/2.0)
			f1 = np.append(f1, noslip_blob_gamma)
			f2 = np.append(f2, noslip_blob_position)
			f3 = np.append(f3, delta_n)
			panel_counter += 1
		body_counter += 1
	# print np.shape(free_vortices1)
	# print f1
	return f1, f2, f3

def plot_tracers(tracers, all_panels, Y, free_stream_velocity, del_t, total_time):
	N, n = get_N_n(all_panels)		
	body_counter, panel_counter = 0, 0
	tracer_trajec = []
	for tracer in tracers:
		tracer_trajec.append([tracer])
	time = 0
	while time < total_time:
		for i in range(len(tracers)):
			velocity = free_stream_velocity
			body_counter = 0
			panel_counter = 0
			for j in range(N):
				if panel_counter > n[body_counter] - 1:
					panel_counter = 0
					body_counter += 1
				[c0,c1] = Y[body_counter][panel_counter]
				velocity += all_panels[body_counter][panel_counter].velo(tracer_trajec[i][-1], c0, c1)
				panel_counter += 1
			final_pos = tracer_trajec[i][-1] + velocity * del_t
			tracer_trajec[i].append(final_pos) 
		time += del_t
	tracer_trajec = np.array(tracer_trajec)
	for i in tracer_trajec:
		plt.plot(i.real, i.imag)
	plt.title("HOW DID IT WORK?")
	plt.show()
##################
######CHANGE TO WORK FOR MULTIPLE BODIES##############
####################################################

def absorb_in_solid(free_vortices_pos, all_panels, free_vortices_gamma):
	body = all_panels[0]
	mask = np.ones(len(free_vortices_pos), dtype = bool)
	gamma_destroyed = 0
	if len(all_panels)>1:
		sys.exit("reflect_off_solid NOT YET BUILD FOR MULTIPLE BODIES")
	for i in range(len(free_vortices_pos)):
		intersections = 0
		for panel in body:
			[v1,v2] = panel
			condition1 = max(v1.imag,v2.imag) > free_vortices_pos[i].imag
			condition2 = (min(v1.real, v2.real) < free_vortices_pos[i].real) & (max(v1.real, v2.real) > free_vortices_pos[i].real)
			if (condition1) and (condition2):
				k = (free_vortices_pos[i].real - v1.real)/((v2-v1).real)
				y_value = (v1 + k*(v2 - v1)).imag
				if y_value > free_vortices_pos[i].imag:
					intersections += 1
		if intersections%2 == 1:
			mask[i] = False
			gamma_destroyed += free_vortices_gamma[i]
	new_vortices_gamma, new_vortices_pos, new_vortices_core_radius = gamma_conserve(all_panels, gamma_destroyed)
	return mask, new_vortices_gamma, new_vortices_pos, new_vortices_core_radius 

def gamma_conserve(all_panels, gamma_destroyed):
	length = 0
	total_panels = 0
	for body in all_panels:
		for panel in body:
			length += abs(panel[0]-panel[1])
			total_panels +=1
	avg_length = length/total_panels
	core_radius = avg_length/np.pi
	new_vortices_pos = []
	for body in all_panels:
		for panel in body:
			new_vortices_pos.append(panel_midpoint(panel) + (core_radius * panel_unit_normal(panel)))
	new_vortices_pos = np.array(new_vortices_pos)
	new_vortices_gamma = np.ones(total_panels) * (gamma_destroyed/float(total_panels))
	new_vortices_core_radius = np.ones(total_panels) * core_radius
	return new_vortices_gamma, new_vortices_pos, new_vortices_core_radius

#####################################################
# @jit
# def reflect_from_circle(free_vortices_pos, circle_radius, circle_center = 0):
# 	for i in range(len(free_vortices_pos)):
# 		z = free_vortices_pos[i] - circle_center 
# 		if abs(z) < circle_radius:
# 			reflected_pos = z/abs(z)
# 			reflected_pos *= 2 * circle_radius - abs(z)
# 			reflected_pos += circle_center
# 			free_vortices_pos[i] = reflected_pos
# 	return 
# @jit
def simulate(all_panels, free_stream_velocity, free_vortices_gamma, free_vortices_pos, free_vortices_core_radius,\
tracers, circulation, body_velocity_fn, viscosity, gamma_max, sim_time, time_step, num_method,name,run_time = 0):
	t_start = get_current_time()
	print t_start
	t_end = t_start + (run_time)
	time = 0
	time_step_counter = 0
	vor_gamma,vor_pos, panel_pos = [], [], []
	x1 = np.linspace(-1.2, 5, 50)
	x2 = np.linspace(-2, 2, 30)

	mesh1 = np.meshgrid(x1, x2)
	mesh2 = np.meshgrid(x2, x2) 
	while get_current_time() < t_end:
	# while time < sim_time:
		body_velocity = body_velocity_fn(time)
		free_vortices_pos, tracers, new_all_panels = convection_step(all_panels, free_stream_velocity, free_vortices_gamma, free_vortices_pos,\
		free_vortices_core_radius, tracers, circulation, body_velocity_fn, time, time_step, num_method)
		
		free_vortices_gamma, free_vortices_pos, free_vortices_core_radius = add_noslip_blobs123(all_panels, free_stream_velocity,\
		free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, circulation, body_velocity)

			
		free_vortices_gamma, free_vortices_pos, free_vortices_core_radius = get_em_drunk( free_vortices_gamma, free_vortices_pos,\
		free_vortices_core_radius, time_step, viscosity, gamma_max)
		mask, new_vortices_gamma, new_vortices_pos, new_vortices_core_radius = absorb_in_solid(free_vortices_pos, all_panels, free_vortices_gamma)
		
		free_vortices_gamma, free_vortices_pos, free_vortices_core_radius = free_vortices_gamma[mask], free_vortices_pos[mask], free_vortices_core_radius[mask]
		print "total_gamma_mask = " + str(free_vortices_gamma.sum())
		free_vortices_gamma = np.append(free_vortices_gamma, new_vortices_gamma)
		free_vortices_pos = np.append(free_vortices_pos, new_vortices_pos)
		free_vortices_core_radius = np.append(free_vortices_core_radius, new_vortices_core_radius)
		
		time += time_step
		time_step_counter += 1
		
		# quiver_plot(all_panels, free_stream_velocity, free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, mesh2, name + "q" + str(time_step_counter))
		all_panels = new_all_panels
		print str(time_step_counter) + "=time step done"
		print "no of vortices =" + str(len(free_vortices_pos))
		panel_pos.append(all_panels)
		vor_pos.append(free_vortices_pos)
		vor_gamma.append(free_vortices_gamma)
		print "total_gamma = " + str(free_vortices_gamma.sum())
		print "------------------------------------------------------"
	print "Simulated" 
	return vor_gamma, vor_pos, panel_pos
def get_current_time():
	return time()
def plot_circle(circle_radius, center = 0, panel_number = 30):
	circle = create_panels_for_circle(center,circle_radius, panel_number)[1]
	plt.plot(circle.real, circle.imag, "k")
def convection_step(all_panels, free_stream_velocity,  free_vortices_gamma, free_vortices_pos, free_vortices_core_radius,\
tracers, circulation, body_velocity_fn, time, time_step, num_method):
	vortice_final_pos, tracer_final_pos, all_panels = num_method(all_panels, free_stream_velocity, free_vortices_gamma,\
	free_vortices_pos, free_vortices_core_radius, tracers, circulation, body_velocity_fn, time, time_step)
	return vortice_final_pos, tracer_final_pos, all_panels

# @jit
def rk2(all_panels, free_stream_velocity,  free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, tracers,\
	circulation, body_velocity_fn, time, time_step):
	vortice_mid_pos, tracer_mid_pos, all_panels_mid = euler(all_panels, free_stream_velocity, free_vortices_gamma,\
	free_vortices_pos, free_vortices_core_radius, tracers, circulation, body_velocity_fn, time, time_step/2.0)
	
	vortice_fin_pos, tracer_fin_pos = euler(all_panels_mid, free_stream_velocity, free_vortices_gamma, vortice_mid_pos,\
	free_vortices_core_radius, tracer_mid_pos, circulation, body_velocity_fn, time+time_step/2., time_step)[0:2]
	all_panels = move_body_panels(all_panels, body_velocity_fn(time + time_step/2.0), time_step)
	vortice_fin_pos += free_vortices_pos - vortice_mid_pos
	tracer_fin_pos += tracers - tracer_mid_pos 
	return vortice_fin_pos, tracer_fin_pos, all_panels

# @jit
def euler(all_panels, free_stream_velocity, free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, tracers,\
	circulation, body_velocity_fn, time, time_step):
	vortice_final_pos, tracer_final_pos = [], []
	body_velocity = body_velocity_fn(time)
	X = solve_panels(all_panels, free_stream_velocity, free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, circulation, body_velocity)
	Y = divide_X_panelwise(all_panels, X)
	for i in range(len(free_vortices_pos)):
		velocity = free_stream_velocity
		velocity += vortices_velo_chorin(np.delete(free_vortices_gamma, i), np.delete(free_vortices_pos, i),np.delete(free_vortices_core_radius, i), free_vortices_pos[i])
		velocity += vel_by_panels(free_vortices_pos[i], all_panels, Y)
		final_pos = free_vortices_pos[i] + velocity * time_step
		vortice_final_pos.append(final_pos)
	for tracer in tracers:
		velocity = free_stream_velocity + vortices_velo_chorin(free_vortices, tracer) + vel_by_panels(tracer, all_panels, Y)
		final_pos = tracer + velocity * time_step
		tracer_final_pos.append(final_pos)
	all_panels = move_body_panels(all_panels, body_velocity, time_step)
	return np.array(vortice_final_pos, dtype = complex), np.array(tracer_final_pos, dtype = complex), all_panels
def I_over_time(vor_gamma, vor_pos):
	I_t = []
	for i in range(len(vor_gamma)):
		I_t.append(I(vor_gamma[i], vor_pos[i]))
	return I_t
def I(free_vortices_gamma, free_vortices_pos, Ro = 1):
	return Ro*(free_vortices_gamma * (free_vortices_pos.imag - 1j * free_vortices_pos.real)).sum()

def F_aero(I_t, delta_t, t_start = 0, avg_no = 1):
	I_t = moving_average(I_t, avg_no)
	time = t_start + (avg_no -1)*delta_t
	F = []
	t = []
	for i in range(len(I_t) - 1):
		F.append((I_t[i] - I_t[i+1])/delta_t)
		t.append(time)
		time += delta_t
	F, t = np.array(F), np.array(t)
	return F, t
@jit
def moving_average(values, n = 3) :
    avg_values = np.cumsum(values, dtype = complex)
    avg_values[n:] = avg_values[n:] - avg_values[:-n]
    return avg_values[n - 1:]/n	
# @jit
def vel_by_panels(z, all_panels, Y):
	velocity = 0
	body_counter = 0
	for body in all_panels:
		panel_counter = 0
		for panel in body:
			[c0,c1] = Y[body_counter][panel_counter]
			velocity += panel_velo(panel,z, c0, c1)
			panel_counter += 1
		body_counter += 1
	return velocity

def move_body_panels(all_panels, body_velocity, time_step):
	new_all_panels = []
	delta_y = body_velocity * time_step
	for panel in all_panels:
		new_all_panels.append(panel + delta_y)
	return new_all_panels
# @jit
def quiver_plot(all_panels, free_stream_velocity, free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, mesh, name):
	x, y = mesh
	z = x + 1j*y
	X = solve_panels(all_panels, free_stream_velocity,free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, circulation)
	Y = divide_X_panelwise(all_panels, X)
	for i in range(len(z)):
		for j in range(len(z[i])):
			vel =  free_stream_velocity + vel_by_panels(z[i][j], all_panels, Y) + vortices_velo_chorin(free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, z[i][j])
			z[i][j] = vel
	plot_panels(all_panels)
	plt.quiver(x, y, z.real, z.imag)
	plt.savefig(name + ".png")
	plt.close()
def countorf_plot(all_panels, free_stream_velocity, free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, mesh, name,limit= 0):
	x, y = mesh
	z = x + 1j*y
	X = solve_panels(all_panels, free_stream_velocity,free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, circulation)
	Y = divide_X_panelwise(all_panels, X)
	for i in range(len(z)):
		for j in range(len(z[i])):
			vel =  free_stream_velocity + vel_by_panels(z[i][j], all_panels, Y) + vortices_velo_chorin(free_vortices_gamma, free_vortices_pos, free_vortices_core_radius, z[i][j])
			z[i][j] = vel
	plt.contourf(x, y, abs(z))
	plot_panels(all_panels)
	plt.colorbar()
	plt.savefig(name + ".png")
	plt.close()
def velo_sin_flap(t, freequency = .1, amplitude = 1, phase = 0):
	omega = 2*np.pi*freequency
	return amplitude * omega * np.cos((omega * t) + phase) * 1j
def zero_velo(t):
	return 0.0

def plot_panels(all_panels):
	for body in all_panels:
		polygon = []
		for i in body:
			polygon.append(i[0])
		polygon.append(body[0][0])
		polygon = np.array(polygon)
		plt.plot(polygon.real, polygon.imag)
	return
def create_panels_flatplate(l, b, dx, AoA = 0):
	vertices = [complex(0, 0)]
	b_dx = int(b//dx)
	l_dx = int(l//dx)
	for i in range(b_dx):
		vertices.append(vertices[-1] + 1j*dx)
	for i in range(l_dx):
		vertices.append(vertices[-1] + dx)
	for i in range(b_dx):
		vertices.append(vertices[-1] - 1j*dx)
	for i in range(l_dx):
		vertices.append(vertices[-1] - dx)
	rad = -AoA*np.pi/180
	vertices = np.array(vertices) * complex(np.cos(rad), np.sin(rad))
	panels = vertices_to_panels(vertices)
	return panels
def create_panels_flat007(l, dx, AoA):
	vertices = [complex(0, 0)]
	l_dx = int(l//dx)
	for i in range(l_dx):
		vertices.append(vertices[-1] + dx)
	panels = []
	for i in range(len(vertices) - 1):
		panels.append(vertices[i:i+2])
	return np.array(panels)
def vertices_to_panels(vertices):
	panels = []
	for i in range(len(vertices) - 1):
		panels.append(vertices[i : i + 2])
	return np.array(panels)

if __name__ == '__main__':

	
	# airfoil = []
	# file_name = 'U2.csv'
	# print file_name
	# file = open(file_name, 'rb')
	# for a in file:

	# 	b = a.rstrip().split(",")
	# 	airfoil.append(complex(float(b[0]), float(b[1])))

	
	# airfoil = np.array(airfoil)
	# AoA = 8
	# AoA *= -1 * np.pi/180.0
	# airfoil = airfoil * complex(np.cos(AoA), np.sin(AoA))
	# plt.plot(v.real, v.imag)
	# plt.show()
	# print len(v)
	# airfoil_panels = []

	# for i in range(len(airfoil)-1):
	# 	airfoil_panels.append([airfoil[i], airfoil[i+1]])
	# i = 0
	# print panel_length([airfoil[i], airfoil[i+1]])
	# while i < len(airfoil):
	# 	print (i+1)%len(airfoil)
	# 	if panel_length([airfoil[i], airfoil[(i+1)%len(airfoil)]]) < .015:
	# 		airfoil_panels.append([airfoil[i], airfoil[(i+2)%len(airfoil)]])
	# 		i +=2
	# 	else:
	# 		airfoil_panels.append(airfoil[i:i+2])
	# 		i += 1
	# all_panels = [airfoil_panels]
	# all_panels = np.array(all_panels)
	# x, length = [], []
	# for i in range(len(all_panels[0])):
	# 	length.append(panel_length(all_panels[0][i]))
	# 	x.append(i)
	# plt.plot(x,length)
	# plt.show() 
	dx = .04
	b = 1*dx
	l = 40*dx
	all_panels = [create_panels_flatplate(l,b,dx, 0)]
	# all_panels = [ create_panels_flat007(30*dx, dx, 0) ]
	############################################################################################
	circulation = np.array([0])
	body_velocity_fn = velo_sin_flap
	# body_velocity_fn = zero_velo
	gamma_max = .01
	Re = 1000
	Ro = 1
	free_stream_velocity = 1
	viscosity = free_stream_velocity*Ro*l/Re
	time_step = .02
	sim_time = 30*time_step
	tracers = []
	tracer = np.array(tracers)
	free_vortices_gamma = np.array([],dtype = float)
	free_vortices_pos = np.array([],dtype = complex)
	free_vortices_core_radius = np.array([],dtype = float)
	num_method = rk2
	

	# name = "0AoA"
	# vor_gamma, vor_pos, panel_pos = simulate(all_panels, free_stream_velocity, free_vortices_gamma, free_vortices_pos, free_vortices_core_radius,tracers, circulation, body_velocity_fn, viscosity, gamma_max, sim_time, time_step, num_method, name,60*60*3.5)

	# print "plotting"
	# for i in range(len(vor_gamma)):
	# 	plot_panels(panel_pos[i])	
	# 	plot_vortex_positions(vor_gamma[i], vor_pos[i], name + str(i))
		
	# I_t = I_over_time(vor_gamma, vor_pos)
	# for no_avg in [3, 10, 15, 25, 50, 100]:
	# 	F, t = F_aero(I_t, time_step, 0, no_avg)
	# 	plt.plot(t, F.real*2/(l*Ro*free_stream_velocity**2))
	# 	plt.title("Cd vs Time (moving point avg with "+str(no_avg)+" time steps)")
	# 	plt.savefig("Cd" + str(no_avg) + name +".png")
	# 	plt.close()
	# 	plt.plot(t, F.imag*2/(l*Ro*free_stream_velocity**2))
	# 	plt.title("Cl vs Time (moving point avg with "+str(no_avg)+" time steps)")
	# 	plt.savefig("Cl" + str(no_avg) + name +".png")
	# 	plt.close()
	
	# F, t = F_aero(I_t, time_step, 0, 1)
	# print "avg force = " + str(np.sum(F)/len(F))
	# print "l = " + str(l)
	# print "AoA = 0 degree"
	# print "Re = " + str(Re)
	# print "Ro = " + str(Ro)






	all_panels = [create_panels_flatplate(l,b,dx, 5)]
	name = "5AoA"
	t_start = get_current_time()
	vor_gamma, vor_pos, panel_pos = simulate(all_panels, free_stream_velocity, free_vortices_gamma, free_vortices_pos, free_vortices_core_radius,tracers, circulation, body_velocity_fn, viscosity, gamma_max, sim_time, time_step, num_method, name,60*60*9)
	print "time = " + str(get_current_time() - t_start)
	print "STOP STOP STOP"
	print "plotting"
	for i in range(len(vor_gamma)):
		plot_panels(panel_pos[i])	
		plot_vortex_positions(vor_gamma[i], vor_pos[i], name + str(i))
		
	I_t = I_over_time(vor_gamma, vor_pos)
	for no_avg in [3, 10, 15, 25, 50, 100]:
		F, t = F_aero(I_t, time_step, 0, no_avg)
		plt.plot(t, F.real)
		plt.title("Drag force vs Time (moving point avg with "+str(no_avg)+" time steps)")
		plt.savefig("Drag" + str(no_avg)+ name + ".png")
		plt.close()
		plt.plot(t, F.imag*2/(l*Ro*free_stream_velocity**2))
		plt.title("Lift force vs Time (moving point avg with "+str(no_avg)+" time steps)")
		plt.savefig("Lift" + str(no_avg) + name + ".png")
		plt.close()
	
	F, t = F_aero(I_t, time_step, 0, 1)
	print "avg force = " + str(np.sum(F)/len(F))
	print "l = " + str(l)
	print "AoA = 5 degree"
	print "Re = " + str(Re)
	print "Ro = " + str(Ro)
	print "time_step = " + str(time_step)
	print "integrator = " + str(num_method)
	print "gamma_max = "  + str(gamma_max)
	print "plate velocity = 2pift*cos(2pift) where f = 0.1" 
