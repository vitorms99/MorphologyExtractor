
import numpy as np
from libc.math cimport atan2, sqrt, cos, sin, M_PI, ceil, floor
from libc.stdlib cimport rand, RAND_MAX

cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef float[:,:] remove_secondary_objects(float[:,:] image, float[:,:] segMask, float idGalaxy):
    cdef: 
        int w, h , i, j
        float[:,:] image_copy
    w, h= len(image[0]), len(image)
    
    image_copy = np.array([[float(0.0) for j in range(w)] for i in range(h)],dtype=np.float32)
    
    for i in range(w):
        for j in range(h):
            if((segMask[j, i] != idGalaxy) & (segMask[j, i] != 0)):
                image_copy[j, i] = 0.0
            else:
                image_copy[j, i] = image[j, i]
    
    return image_copy

#########################################################################
#    interpolate_ellipse
#########################################################################
# Objetivo:
#    Rotacionar e interpolar a grade nos pontos com o valor 0.0
# Entrada:
#    image - matriz de entrada - np.array(list of list)
#    nRot - numero de rotacoes - int

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef float[:,:] interpolate_ellipse(float[:,:] image, float angle):
    cdef:  
        float[:,:] image_copy
        float[:] delta_radius, 
        float sc, rho, mean, max_rad, min_rad
        float auxx, auxy
        int nx, ny, d_rho, center_x, center_y
        int w, h, i, j, same_ellipse

    w, h = len(image[0]), len(image)
    #center_x, center_y = int(ceil(w/2.0)), int(ceil(h/2.0))
    center_x, center_y = int(floor(w/2.0)), int(floor(h/2.0))
    max_rad = float(h/2.0)
    min_rad = max_rad/1.5

    image_copy = np.array([[float(0.0) for j in range(w)] for i in range(h)], dtype=np.float32)
    mask = np.array([[float(0.0) for j in range(w)] for i in range(h)], dtype=np.float32)
    delta_radius = np.array([0.0 for i in range(len(image_copy))], dtype=np.float32)
    
    for i in range(0, len(image_copy)):
        delta_radius[i] = float(2.0)*float(i)*M_PI/len(image_copy)-M_PI
        
    for i in range(w):
        for j in range(h):
            if(image[j, i]!=0.0):
                image_copy[j, i] = image[j,i]
                mask[j,i] = 0.0
                continue
            mask[j,i] = 1.0
            sc = find_scale(i, j, center_x, center_y, angle, max_rad, min_rad)
            rho = atan2(j-center_x, i-center_y)
            same_ellipse = 0 
            mean = 0.0
            lstPts = []
            
            for d_rho in range(len(delta_radius)):
                auxx, auxy = find_point(sc, rho+delta_radius[d_rho], max_rad, min_rad, center_x, center_y, angle)
                if (not np.isnan(auxx) and not np.isnan(auxy)):
                    nx, ny = int(auxx), int(auxy)
                else:
                    nx, ny = -1, -1
                if (nx>=0) & (ny>=0) & (nx<w) & (ny<h):
                    if (image[ny, nx]!= 0.0):
                        same_ellipse = same_ellipse + 1
                        lstPts.append(image[ny, nx])
            
            if same_ellipse < 1:
                image_copy[j, i] = np.average(image)
                continue
            
            if(len(lstPts)>0):
                median = np.median(lstPts)
                std = 0.743*(np.quantile(lstPts, .75) - np.quantile(lstPts, .25))
                
                image_copy[j, i] = np.random.normal(loc = median, scale = std)
    
    return image_copy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float find_scale(float x, float y, int center_x, int center_y, float angle, float max_rad, float min_rad):
    cdef float dx = x - float(center_x)
    cdef float dy = y - float(center_y)
    cdef float part1, part2

    part1 = (dx * cos(angle) + dy * sin(angle)) / (max_rad)
    part2 = (dx * sin(angle) - dy * cos(angle)) / (min_rad)

    return sqrt((part1 * part1) + (part2 * part2))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef tuple find_point(float scale, float rho, float max_rad, float min_rad, int center_x, int center_y, float angle):
    x, y = scale * max_rad * cos(rho), scale * min_rad * sin(rho)
    x2, y2 = center_x + x * cos(angle) - y * sin(angle), center_y + x * sin(angle) + y * cos(angle)
    return x2, y2
