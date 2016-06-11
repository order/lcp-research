import numpy as np
from mdp.transitions import new_slope
from discrete import make_points

import scipy.integrate as integrate

import matplotlib.pyplot as plt

import time

def norm(x,y,z):
    return np.sqrt(x*x + y*y + z*z)

def normalize(x,y,z):
    Z = norm(x,y,z)
    if (Z == 0):
        return (np.zeros(x.shape),
                np.zeros(y.shape),
                np.zeros(z.shape))
    return (x/Z,y/Z,z/Z)
    

def triangle_wave(x,P,A):
    # P is the period,
    # A is the amplitude
    x = x/P
    return A*(2 * np.abs(2* (x - np.floor(x+0.5))) - 1)

def soft_threshold(x,thres):
    return np.sign(x) * np.maximum(0,np.abs(x)-thres)

def x_der(x):
    P = 8
    A = 2
    return triangle_wave(x - P/4,P,A)

def y_der(y):
    P = 8
    A = 0.5
    return triangle_wave(y - P/4,P,A)

def grad_map(x,y):
    return (x_der(x),y_der(y))

def tangent_map_angle(x,y,a):
    fx = np.cos(a)*np.ones(x.shape)
    fy = np.sin(a)*np.ones(y.shape)
    
    Gx,Gy = grad_map(x,y)
    fz = fx * Gx + fy * Gy
    
    return normalize(fx,fy,fz)

def tangent_map_vector(x,y,u,v):
    Gx,Gy = grad_map(x,y)
    w = u * Gx + v * Gy    
    return normalize(u,v,w)    
    
def normal_map(x,y):
    Gx,Gy = grad_map(x,y)
    return normalize(-Gx,-Gy,np.ones(Gx.size))

def total_der_map(x,y):
    Gx,Gy = grad_map(x,y)
    return Gx + Gy

def height(x,y,x0,y0):
    Ix = integrate.quad(x_der,x0,x)[0]
    Iy = integrate.quad(y_der,y0,y)[0]
    return Ix+Iy

def height_map(x,y,x0,y0):
    # For scattered data.
    (N,) = x.shape
    assert((N,) == y.shape)

    H = np.empty(N)
    for i in xrange(N):
        H[i] = height(x[i],y[i],x0,y0)
    return H

# Grid data
def height_map_mesh(x_desc,y_desc):
    (lox,hix,nx) = x_desc # nx is vertex count, not cell
    (loy,hiy,ny) = y_desc

    H = np.empty((nx,ny))
    gridx = np.linspace(lox,hix,nx)
    gridy = np.linspace(loy,hiy,ny)

    for i in xrange(nx):
        if i == 0:
            H[i,0] = 0
        else:
            res = integrate.quad(x_der,gridx[i-1],gridx[i])
            H[i,0] = H[i-1,0] + res[0]
            
        for j in xrange(1,ny):
            res = integrate.quad(y_der,gridy[j-1],gridy[j])
            Intx = H[i,j-1] + res[0]
            if i == 0:
                H[i,j] = Intx
            else:
                res = integrate.quad(x_der,gridx[i-1],gridx[i])
                Inty = H[i-1,j] + res[0]
                H[i,j] = (Intx + Inty) / 2.0    
               
    return H


def control_force(x,y,a,u):
    # Apply force tangent to surface at angle a
    (cx,cy,cz) = tangent_map_angle(x,y,a)
    return (u*cx,u*cy,u*cz)

def total_force(x,y,vx,vy,a,u):
    (cfx,cfy,cfz) = control_force(x,y,a,u)
    grav = 9.806

    # Gravity + control
    tx = cfx
    ty = cfy
    tz = cfz-grav

    # Add approx friction
    mu = 0.0 # Fairly low coefficient
    Tx,Ty,Tz,Fx,Fy,Fz = friction(x,y,vx,vy,tx,ty,tz,mu)

    # Return Gravity + control + friction
    return Tx,Ty,Tz,Fx,Fy,Fz

def friction(x,y,vx,vy,fx,fy,fz,mu):
    # Break into tangent and normal 
    (Tx,Ty,Tz,Nx,Ny,Nz) = decompose_normal(x,y,fx,fy,fz)
    p = norm(Nx,Ny,Nz) # Normal force magnitude

    # Friction is in the opposite direction as the motion
    (fx,fy,fz) = p*mu*tangent_map_vector(x,y,-vx,-vy)

    return (Tx,Ty,Tz,fx,fy,fz)
    
def inner_product(x,y,z,u,v,w):
    return x*u + y*v + z * w


def decompose_normal(x,y,fx,fy,fz):
    Nx,Ny,Nz = normal_map(x,y)

    p = inner_product(fx,fy,fz,Nx,Ny,Nz)
    return fx - p*Nx,fy - p*Ny, fz - p*Nz,p*Nx,p*Ny,p*Nz

def quasi_quiver(ax,x,y,z,u,v,w,**kwargs):
    (N,) = x.shape
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.set_zlim(-2,6)
    for i in xrange(N):
        plt.plot([x[i],x[i]+u[i]],
                 [y[i],y[i]+v[i]],
                 [z[i],z[i]+w[i]],'-k')
    plt.show()
    


if __name__ == '__main__':
    
    Nx = 33
    Ny = 33
    Tx = 33 # Resolution for terrain
    Ty = 33
    xdesc  = (-2,6,Nx)
    ydesc  = (-2,6,Ny)
    txdesc = (-2,6,Tx)
    tydesc = (-2,6,Ty) 
    (P,(X,Y)) = make_points([np.linspace(*xdesc),np.linspace(*ydesc)],True)
    (_,(TX,TY)) = make_points([np.linspace(*txdesc),np.linspace(*tydesc)],True)

    x = P[:,0]
    y = P[:,1]
    
    H = height_map_mesh(xdesc,ydesc)
    h = H.flatten()
    TH = height_map_mesh(txdesc,tydesc)
    
    #(Nx,Ny,Nz) = normal_map(x,y)


    # Basic surface
    fig = plt.figure()
    ax = plt.subplot(1,2,1,projection='3d')
    ax.set_zlim(-2,6)

    if True:
        M = np.max(h)
        m = np.min(h)
        
        ax.plot_surface(TX,TY,TH,rstride=1, cstride=1, cmap='terrain',
                        clim=[m - 0.3*(M-m),M],
                        linewidth=0, antialiased=False)

    h = H.flatten()

    # Total force
    if False:
        (fx,fy,fz) = total_force(x,y,np.pi,1)
        quasi_quiver(ax,x,y,h,0.05*fx,0.05*fy,0.05*fz)
    if False:
        (fx,fy,fz) = total_force(TX.flatten(),TY.flatten(),np.pi,0)
        f_img = np.reshape(norm(fx,fy,fy),(Tx,Ty))
        ax.plot_surface(TX,TY,f_img,rstride=1, cstride=1, cmap='plasma',
                        linewidth=0, antialiased=False)
        
    # Normal vectors
    if False:
        (Nx,Ny,Nz) = normal_map(x,y)
        quiv = ax.quiver(x,y,h,Nx,Ny,Nz,
                         pivot='tail',arrow_length_ratio=0,length=0.2,alpha=0.25)
        quiv.set_color('r')


    if True:
        N = 1
        x = np.array([6.0])
        y = np.array([1.0])
        z = height_map(x,y,-2,-2)

        vx = np.array([0.0])
        vy = np.array([0.0])
        
        I = 500
        P = np.empty((I,5))
        t = 0.005
        d = 1e-4
        
        for i in xrange(I):
            P[i,0] = x
            P[i,1] = y
            P[i,2] = z
            P[i,3] = vx
            P[i,4] = vy

            vx *= (1-d)
            vy *= (1-d)

            (tx,ty,tz,fx,fy,fz) = total_force(x,y,vx,vy,np.pi,2)

            x += t*vx
            y += t*vy
            x = np.maximum(-2,np.minimum(6,x))
            y = np.maximum(-2,np.minimum(6,y))

            z = height_map(x,y,-2,-2)

            vx += t*tx + t*fx
            vy += t*ty + t*fy

    plt.plot(P[:,0],P[:,1],P[:,2],'k-x')
    
    plt.subplot(1,2,2)
    plt.plot(P[:,:5],'-x')
    plt.legend(['x','y','z','vx','vy'])
    plt.show()
    
    

    
