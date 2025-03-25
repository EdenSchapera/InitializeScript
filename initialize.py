# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 00:22:21 2025

@author: Eden Schapera




1D Brio-Wu Shock tube parameters from: https://www.astro.princeton.edu/~jstone/Athena/tests/brio-wu/Brio-Wu.html
and originally documented in Brio, M. & C.C. Wu, "An Upwind Differencing Scheme for the Equations of Ideal Magnetohydrodynamics",
Journal of Computational Physics, 75, 400-422 (1988).

"""

import numpy as np


def initialize_mhd( nx = 400, # generic 400x400x400 3d grid from -1 to 1 in all dimensions
                    ny = 400,
                    nz = 400,

                    x_min = -1,
                    x_max = 1,
                
                    y_min = -1,
                    y_max = 1,

                    z_min = -1,
                    z_max = 1,

                    max_steps = 10000,
                    final_time = 0.2,

                    cfl = 0.4, # note, generally should be computed as delT <= CFL <= min(delX,delY,delZ) / max(lambda_max)
                                # this is just an estimate but seems to be in the right range for our sim

                    gamma = 0, # ratio of specific heats
                    Bx_const = 0, # constant Bx (for 1d sims)
                    test_case = '' # test case flag for specific models of interest
                    ):
    
    
    ### Parameters
    ''' 
    Initializes 1d MHD variables 
    
    
    Inputs (Optional, uses default values if none provided)
    
    
    nx                      : number of grid points                 : 400
    x_min                   : minimum x value (m)                   : -1 
    x_max                   : maximum x value (m)                   : 1

    ny                      : number of grid points                 : 0
    y_min                   : minimum y value (m)                   : 0 
    y_max                   : maximum y value (m)                   : 0

    nz                      : number of grid points                 : 0
    z_min                   : minimum y value (m)                   : 0
    z_max                   : maximum y value (m)                   : 0



    gamma                   : ratio of specific heats               : 0
    max_steps                : maximum number of simulation steps    : 10000
    final_time              : final simulation time (seconds)       : 0.2
    cfl                     : CFL number                            : 0.04
    Bx_const                : X component of B (constant)           : 0.75

    test_case               : flag for cases of interest            : 'shock_tube'
                            : current allowed values                : '', 'shock_tube'
    
    Outputs:
        
    xcoord                  : nx array of x grid points
    ycoord                  : nx array of y grid points
    zcoord                  : nx array of z grid points

    rho, Vx, ... Bz, E      : nx,ny,nz numpy arrays for each conserved variable
    
    
    dx                      : grid spacing
    dy
    dz


    dt                      : time step
    nstep                   : initial step
    max_step                : maximum number of steps for the simulation
    final_time              : end time
    gamma                   : ratio of specific heats

    cfl                     : cfl number
    case                    : case identifier
    
    ''' 


    # Parameter initialization for the shock tube case
    if test_case == 'shock_tube':
        print("Assigning initial values for 1D Brio Wu shock tube")

        # force other dimensions to 1
        ny = 1
        nz = 1
        
        # if no gamma set 
        if gamma == 0:
            print("Gamma = 2.0")
            gamma = 2.0

        # if no constant bx set
        if Bx_const == 0:
            print("Bx = 0.75")
            Bx_const = 0.75

        # if no x coordinate points
        if nx == 0:
            print("400 Coordinate Points")
            nx = 400
        
        # if no values for xmin or max
        if x_min == 0 and x_max == 0:
            print("Xmin,Xmax = {-1,1}")
            x_min = -1.0
            x_max = 1.0



    
    # Error handling if someone tries to create a zero dimensional system
    if nx == 0 and ny == 0 and nz == 0:
        print("Error: number of grid points for nx, ny, nz cannot all be zero")
        return None

   
    ''' 
    
    Create coordinate grid for x, y, and z

    '''

    if nx >= 2:
        # define simulation grid with nx points
        # stretch between xmin and xmax
        dx = (x_max - x_min) / (nx-1)
        xcoord = np.linspace(x_min,x_max,nx)
    else: 
        dx = 0
        xcoord = np.array([0.0])  # trivial coordinate    
    
    
    if ny >= 2:
        # define simulation grid with ny points
        # stretch between ymin and xmax
        dy = (y_max - y_min) / (ny-1)
        ycoord = np.linspace(y_min,y_max,ny)
    else: 
        dy = 0
        ycoord = np.array([0.0])

    
    if nz >= 2:
        # define simulation grid with nz points
        # stretch between zmin and zmax
        dz = (z_max - z_min) / (nz-1)
        zcoord = np.linspace(z_min,z_max,nz)
    else: 
        dz = 0
        zcoord = np.array([0.0])


    

    # Allocate arrays for conserved variables we are interested in
    # each variable has a nx points


    # Create meshgrid so that X, Y, Z have shape (nx, ny, nz)
    # (even if ny=1 or nz=1, the shapes will be consistent)
    X, Y, Z = np.meshgrid(xcoord, ycoord, zcoord, indexing='ij')
    

    # density
    rho = np.zeros((nx,ny,nz))
    # velocities
    vx = np.zeros((nx,ny,nz))
    vy = np.zeros((nx,ny,nz))
    vz = np.zeros((nx,ny,nz))
    
    # magnetic field
    Bx = np.zeros((nx,ny,nz))
    By = np.zeros((nx,ny,nz))
    Bz = np.zeros((nx,ny,nz))
    
    # energy
    E = np.zeros((nx,ny,nz))
    

    # If it's the Brio-Wu shock-tube test
    if test_case == 'shock_tube':
        # Set Bx to a constant everywhere
        Bx[:] = Bx_const

        # Mask for points where x < 0
        left_mask  = (X < 0.0)
        right_mask = ~left_mask

        # Assign left side
        rho[left_mask] = 1.0
        By[left_mask]  = 1.0
        p_left         = 1.0

        # Assign right side
        rho[right_mask] = 0.125
        By[right_mask]  = -1.0
        p_right         = 0.1

        # Make a pressure array matching left/right masks
        p = np.zeros((nx, ny, nz))
        p[left_mask]  = p_left
        p[right_mask] = p_right
        
        # Compute squared magnitudes
        B_sq = Bx**2 + By**2 + Bz**2
        v_sq = vx**2 + vy**2 + vz**2
        
        # Total energy: E = p/(gamma - 1) + 0.5 * rho * v^2 + 0.5 * B^2
        E[:] = p / (gamma - 1.0) + 0.5 * rho * v_sq + 0.5 * B_sq
        
        print("Left state initialized as (rho, vx, vy, vz, Bx, By, Bz, p) = "
              "[1, 0, 0, 0, {:.2f}, 1, 0, 1]".format(Bx_const))
        print("Right state initialized as (rho, vx, vy, vz, Bx, By, Bz, p) = "
              "[0.125, 0, 0, 0, {:.2f}, -1, 0, 0.1]".format(Bx_const))

    else:
        # Generic setup: everything remains zero (as already initialized)
        # If you'd like something else as a default, you can vectorize similarly.
        pass
    
    
    # initialize time step variables
    dt = 0 # updated each time step from CFL
    time = 0 
    nstep = 0
    
    
    # return all initialization variables as a dictionary
    
    return {
    "xcoord"    : xcoord,
    "ycoord"    : ycoord,
    "zcoord"    : zcoord,

    "rho"       : rho,
    "Vx"        : vx,
    "Vy"        : vy,
    "Vz"        : vz,
    
    "Bx"        : Bx,
    "By"        : By,
    "Bz"        : Bz,

    "E"         : E,

    "dx"        : dx,
    "dy"        : dy,
    "dz"        : dz,

    "dt"        : dt,
    "time"      : time,
    "nstep"     : nstep,
    "max_steps" : max_steps,
    "final_time": final_time,

    "gamma"     : gamma,

    "cfl"       : cfl,

    "case"      : test_case      
    }
               

# unit test funciton
if __name__ == "__main__":
    # Example usage
    params = initialize_mhd(test_case = 'shock_tube')
    print("case:", params["case"])
    print("Grid shape:", params["xcoord"].shape)
    print("Initial density (center):", params["rho"][175:225])           
           

           
           
       
   
    

