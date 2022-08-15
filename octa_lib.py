import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from skimage.draw import line
from skimage.draw import polygon_perimeter
from skimage.draw import polygon
#from skimage import data
from skimage.feature import match_template
from skimage import feature

# for reading images
from pims import pipeline
from pims import Frame
import pims


colorlist = ('blue','green','red','orange','cyan','brown','yellow','magenta','violet','gray')
lcolorlist = len(colorlist)


# face list for an octahedron by hand 
octa_faces = np.array([
       [3, 4, 1, 0],
       [3, 4, 2, 1],
       [3, 4, 3, 2],
       [3, 4, 0, 3],
       [3, 0, 1, 5],
       [3, 1, 2, 5],
       [3, 2, 3, 5],
       [3, 3, 0, 5]])
#The three in front of each face is the number of vertices in the face 
# the other numbers are indices of vertices.  There are 6 vertices so these numbers range from 0 to 5

# list of points for an octahedron by hand
octa_vertices = np.array([
                 [-0.70710677, -0.70710677,  0.        ],
                 [ 0.70710677, -0.70710677,  0.        ],
                 [ 0.70710677,  0.70710677,  0.        ],
                 [-0.70710677,  0.70710677,  0.        ],
                 [ 0.        ,  0.        , -1.        ],
                 [ 0.        ,  0.        ,  1.        ]], dtype=float)
#these are the vertices of the octahedron, there are 6
# 4 of them are on the xy plane
# 2 of them are at +-1 in z


cube_vertices = np.array([
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5],
            [-0.5,  0.5,  0.5],
            [-0.5,  0.5, -0.5],
            [ 0.5, -0.5, -0.5],
            [ 0.5,  0.5, -0.5],
            [ 0.5,  0.5,  0.5],
            [ 0.5, -0.5,  0.5]], dtype=float)
            
cube_faces = np.array([
    [4, 0, 1, 2, 3],
    [4, 4, 5, 6, 7],
    [4, 0, 4, 7, 1],
    [4, 3, 2, 6, 5],
    [4, 0, 3, 5, 4],
    [4, 1, 7, 6, 2]])
    
tet_vertices =  np.array([
        [ 0.57735026,  0.57735026,  0.57735026],
        [-0.57735026,  0.57735026, -0.57735026],
        [ 0.57735026, -0.57735026, -0.57735026],
        [-0.57735026, -0.57735026,  0.57735026]], dtype=float)
        
tet_faces = np.array([
       [3, 0, 1, 2],
       [3, 1, 3, 2],
       [3, 0, 2, 3],
       [3, 0, 3, 1]])


######################################
# routines for vectors 

# length of a vector
def len_vec(A):
    len_A = np.sqrt(np.sum(A**2))
    return len_A

# return cross product of 2 vectors
def cross_prod(A,B):
    cx = A[1]*B[2] - A[2]*B[1]
    cy = A[2]*B[0] - A[0]*B[2]
    cz = A[0]*B[1] - A[1]*B[0]
    return np.array([cx,cy,cz])

# return projected vector into plane perpendicular to nvec
# ee is a vector, nvec is a vector 
def proj_vec(ee,nvec):
    len_n = len_vec(nvec)
    nvec_hat = nvec/len_n      # make sure normalized to length 1
    a = np.sum(nvec_hat * ee)  #dot product of ee and nvec_hat
    ee_perp = ee - a*nvec_hat
    return ee_perp


# routines for lists of vertices and faces
# compute normals from a list of vertices and faces
# this routine is used to determine which faces are visible
# this routine uses edge_vfn(),cros_prod(),len_vec()
# this routine calles edges_vfn() which should work for polygon faces
def compute_normals(vertices,faces):
    nfaces = faces.shape[0]
    normals = np.zeros((nfaces,3))
    for i in range(nfaces):
        face = faces[i,:]
        e1, e2, e3 = edges_vfn(vertices,face) # get edges
        # (if face is a polygon these will be segments in plane)
        norm_i = cross_prod(e1,e2)
        norm_i /= len_vec(norm_i)  #make a unit vector
        normals[i,:] = norm_i
    return normals


# find viewable faces from a list of vertices and faces,
# and a viewing vector pointing to viewer
# result is an array that has same length as the number of faces
# 1 is given in array if face is viewable, otherwise is 0
# this routine uses the above compute_normals() routine
# this routine should work for polygon faces
def face_viewable_vfn(vertices,faces,nvec):
    nfaces = faces.shape[0]
    isview = np.zeros(nfaces)
    normals = compute_normals(vertices,faces)
    for i in range(nfaces):
        norm_i = normals[i,:]
        dotprod = np.sum(norm_i*nvec)
        if (dotprod > 0):
            isview[i] = 1
    return isview


# return three edge vectors from a triangular face
# uses face indexes from 1 to 3
# using a list of vertices and faces
# here face is an array that contains indices of vertices
# if called on a polygon, this will return three
# segments in the plane of polygon, and cross prod of
# first two edges should give the polygon normal
def edges_vfn(vertices,face):
    i1 = face[1] # first vertex
    i2 = face[2]
    i3 = face[3]
    e21 = vertices[i2,:] - vertices[i1,:]  # 2-1
    e32 = vertices[i3,:] - vertices[i2,:]  # 3-2
    e13 = vertices[i1,:] - vertices[i3,:]  # 1-3
    return e21, e32, e13
    
# return an array of edge vectors for a single face
# the number of edges depends on what type of polygon
# the face is
def edges_poly(vertices,face):
    nv = face[0]  # number of vertices in each polygon
    edge_arr = np.zeros((nedges,3))
    for k in range(nv):
        kp1 = (k+1)%nv
        i2 = face[kp1+1]
        i1 = face[k+1]
        ee = vertices[i2,:] - vertices[i1,:]
        edge_arr[k,:] = ee
    return edge_arr


    
# find visible edges from a shape that has polygon faces
# returns an array of edges
# the array is an array where each element is 2 indices of vertices
def find_visible_edges_poly(vertices,faces):
    nvec = np.array([0.,0.,1.])  # viewer direction
    nfaces = faces.shape[0]
    edge_list = np.zeros((0,2),dtype=int)
    isview_list = face_viewable_vfn(vertices,faces,nvec) # viewable faces
    for i in range(nfaces):
        if (isview_list[i] > 0):  # only consider edges of visible faces
            face = faces[i,:]
            nv = face[0]  # number of vertices in face
            for k in range(nv):  # loop over all edges in polygon
                kp1 = (k+1)%nv
                j1 = face[k+1]  # indices of 2 vertices
                j2 = face[kp1+1]
                #print(j1,j2)
                #v1 = vertices[i1,:]  # vertices of edge
                #v2 = vertices[i2,:]
                edge_exists = 0
                for m in range(edge_list.shape[0]):
                    if (j1 == edge_list[m,0]) and (j2 == edge_list[m,1]):
                        edge_exists = 1
                        break
                    if (j2 == edge_list[m,0]) and (j1 == edge_list[m,1]):
                        edge_exists = 1
                        break
                if (edge_exists ==0):
                    edge_list = np.append(edge_list,[[j1,j2]],axis=0)
        
    return edge_list
    
 
# test for this routine!
#rot = R.random()
#vertices = rot.apply(cube_vertices)
#edge_list = find_visible_edges_poly(vertices,cube_faces)
#plt_edges(edge_list,vertices)


###################################################
# plot a list of edges in a figure
# if you give this routine only visible faces then only visible
# ones will be plotted!
def plt_edges(edge_list,vertices):
    fig,ax = plt.subplots(1,1,figsize=(3,3),dpi=100)
    ax.set_aspect(1)
    ax.set_xlim((-1,1))
    ax.set_ylim((-1,1))
    nedges = edge_list.shape[0]
    for k in range(nedges):
        i1 = edge_list[k,0]  #indices of vertices
        i2 = edge_list[k,1]
        #print(i1,i2)
        v1 = np.squeeze(vertices[i1,:])
        v2 = np.squeeze(vertices[i2,:])
        x1 = v1[0]
        x2 = v2[0]
        y1 = v1[1]
        y2 = v2[1]
        ax.plot([x1,x2],[y1,y2],'r-',lw=2)
    
# plot an edge list on ax (axis for plotting)
# with shift by r, c and using scale_fac to expand model
# and y flipped
# with color ecolor
# inputs:
#   edge_list: array of edges (pairs of vertice indices) to plot
#   vertices:  vertices of model
#   ax:  plotting axis
#   scale_fac:  expand model by this factor
#   r,c  to set location of center of mass in pixels on image
#   ecolor:  what color lines
#   lw: line width
#   doflip:  if 1 then invert y
# this routine will work with models that have polygon faces
def plt_edges_s_ax(edge_list,vertices,ax,scale_fac,r,c,ecolor,lw,doflip):
    #fig,ax = plt.subplots(1,1,figsize=(3,3),dpi=100)
    #ax.set_aspect(1)
    #ax.set_xlim((-1,1))
    #ax.set_ylim((-1,1))
    if (doflip==1):
        yf = -1
    else:
        yf = 1
    nedges = edge_list.shape[0]
    for k in range(nedges):
        i1 = edge_list[k,0]  #indices of vertices
        i2 = edge_list[k,1]
        #print(i1,i2)
        v1 = np.squeeze(vertices[i1,:])
        v2 = np.squeeze(vertices[i2,:])
        x1 = v1[0]*scale_fac + r
        x2 = v2[0]*scale_fac + r
        y1 = yf*v1[1]*scale_fac + c  # note y flip here!!!!!!! if yf=-1
        y2 = yf*v2[1]*scale_fac + c
        ax.plot([x1,x2],[y1,y2],'-',lw=lw,alpha=0.5,color=ecolor)
        
        

# return a rotation transformation function using axis angle form
# here theta,phi give you axis nhat to rotate about
# theta_rot is angle to rotate about this axis
# here R is a rotation as in scipy.spatial.transform
# return the rotation operator
def mkrot(theta,phi,theta_rot):
    nhat = np.array((np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta))) # unit vector direction
    r=R.from_rotvec(theta_rot * nhat)  # using rotation vector
    #print('mkrot: nhat =', nhat)
    # r is now a rotation transform function
    #   you can rotate a list of vectors with r.apply(vectors)
    return r


# assume that viewer is looking along +z axis , plot faces with different colors
# only plot visible faces
# use color list to show each face in a different color
# returns matplotlib axis class
# this routine uses face_viewable_vfn()
# works for polygon faces
# calls: face_viewable_vfn()
def plt_vis_faces(vertices,faces):
    nvec = np.array([0.,0.,1.])
    isview_list = face_viewable_vfn(vertices,faces,nvec)
    nfaces = faces.shape[0]
    fig,ax = plt.subplots(1,1,figsize=(3,3))
    ax.set_aspect(1)
    for i in range(nfaces):  #loop over faces!
        if (isview_list[i] > 0):  # only plot visible faces
            face = faces[i,:]
            nv = face[0]  #number of vertices
            xvec = np.zeros(nv) # x values of each vertex
            yvec = np.zeros(nv) # y values of each vertex
            for k in range(nv):  # loop over vertices in face
                j = face[k+1]  # index of vertex
                pv = vertices[j,:]  # vertex
                xvec[k] = pv[0] # x value
                yvec[k] = pv[1] # y value
            
            ax.fill(xvec,yvec,colorlist[i%lcolorlist])  # fill in the polygon
    return ax
 
# test:
#rot = R.random()
#vertices = rot.apply(cube_vertices)
#edge_list = find_visible_edges_poly(vertices,cube_faces)
#plt_vis_faces(vertices,cube_faces)


# assume that viewer is looking along +z axis, plot face edges with black
# only plot visible faces
# returns matplotlib axis class
# this routine uses face_viewable_vfn(), find_visible_edges_poly()
# works for polygon faces
def plt_vis_faces_edges(vertices,faces):
    edge_list= find_visible_edges_poly(vertices,faces)
    nvec = np.array([0.,0.,1.])
    isview_list = face_viewable_vfn(vertices,faces,nvec)
    nfaces = faces.shape[0]
    fig,ax = plt.subplots(1,1,figsize=(3,3))
    ax.set_aspect(1)
    scale_fac = 1; r=0; c=0; ecolor='black'; doflip=0
    plt_edges_s_ax(edge_list,vertices,ax,scale_fac,r,c,ecolor,2,doflip)
    #for i in range(nfaces):
    #    if (isview_list[i] > 0):  # only plot visible faces
    #        i1 = faces[i,1]  # the indices of the vertices of the face
    #        i2 = faces[i,2]
    #        i3 = faces[i,3]
    #        p1 = vertices[i1,:]  # the vertices of the face
    #        p2 = vertices[i2,:]
    #        p3 = vertices[i3,:]
    #        k=0;  x = np.array([p1[k],p2[k],p3[k],p1[k]])  # x,y values only
    #        k=1;  y = np.array([p1[k],p2[k],p3[k],p1[k]])
    #        ax.plot(x,y,'k',lw=3)
    return ax


# Display all rotations in the list
# inputs:
#   rot_list: a list of rotations
#   vertices,faces  a shape
# displays edges but only of visible faces
def disp_rot_list(rot_list,vertices,faces):
    nw = int(np.sqrt(len(rot_list)) ) + 1
    fig,axarr = plt.subplots(nw,nw,figsize=(5,5),sharex = True,sharey=True)
    plt.subplots_adjust(hspace=0,wspace=0)
    axarr_lin = axarr.reshape(-1)
    axarr_lin[0].set_aspect(1.)
    #axarr_lin[0].set_xlim=([-1.,1.])
    #axarr_lin[0].set_ylim=([-1.,1.])
    
    nvec = np.array([0.,0.,1.])
    nfaces = faces.shape[0]
    f=0
    for rot in rot_list:
        new_vertices = rot.apply(vertices)
        isview_list = face_viewable_vfn(new_vertices,faces,nvec)
        
        for i in range(nfaces):
            if (isview_list[i] > 0):  # only plot visible faces
                i1 = faces[i,1]  # the indices of the vertices of the face
                i2 = faces[i,2]
                i3 = faces[i,3]
                p1 = new_vertices[i1,:]  # the vertices of the face
                p2 = new_vertices[i2,:]
                p3 = new_vertices[i3,:]
                k=0;  x = np.array([p1[k],p2[k],p3[k],p1[k]])  # x,y values only
                k=1;  y = np.array([p1[k],p2[k],p3[k],p1[k]])
                if (nw < 7):
                    axarr_lin[f].plot(x,y,'k',lw=3)
                else:
                    axarr_lin[f].plot(x,y,'k',lw=2)
        f+= 1
    
    
# Make binary images of edges of all rotations in the list
# inputs:
#   rot_list: a list of rotations
#   vertices,faces

#from skimage.draw import polygon
#https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.polygon


# Make binary images (templates) of edges of all rotations in the list
# inputs:
#   rot_list: a list of rotations
#   vertices,faces: the model shape
#   scalef_fac: what to multiply the model by
#   width: line half-width in pixels
# use the edge_list to draw visible edges on an image
# put a width for the line using skimage routine fill polygon
# width parameter is actually half width of rectangle
# calls:
#   find_visible_edges_poly()
#   skimage polygon
def big_bin_rot_list3(rot_list,vertices,faces,scale_fac,width):
    f=0
    nr = len(rot_list)  # number of rotations
    nc = int(scale_fac*2.1) # size of images we want to return
    center = int(nc/2)
    img_arr = np.zeros((nr,nc,nc),dtype=np.uint8)
    epsilon = 0.1
    for rot in rot_list:
        new_vertices = rot.apply(vertices)
        edge_list = find_visible_edges_poly(new_vertices,faces)
        #isview_list = face_viewable_vfn(new_vertices,faces,nvec)
        nedges = edge_list.shape[0]
        
        for i in range(nedges):
            i1 = edge_list[i,0] # the indices of the vertices of the edge
            i2 = edge_list[i,1] #
            p1 = np.squeeze(new_vertices[i1,0:2]) # the vertices of the edge, xy values only
            p2 = np.squeeze(new_vertices[i2,0:2])
            
            k=0;  x = np.array([p1[k],p2[k]])  # x,y values only
            k=1;  y = np.array([p1[k],p2[k]])
            x *= scale_fac  # enlarge to actual size of dice
            y *= scale_fac  #
            x+= center
            y+= center
            if (width>0):
                nhat = (p2-p1)/np.sqrt(np.sum((p2-p1)**2) + epsilon)*width
                # vector parallel to edge with length 'width'
                sx = nhat[1]; sy = -nhat[0] # vector perpendicular to edge with length 'width'
                x1 = int(x[0]+sx); x2 = int(x[1]+sx); x3 = int(x[1]-sx); x4 = int(x[0]-sx)
                y1 = int(y[0]+sy); y2 = int(y[1]+sy); y3 = int(y[1]-sy); y4 = int(y[0]-sy)
                xlist = (x1,x2,x3,x4);
                ylist = (nc-y1,nc-y2,nc-y3,nc-y4)
                # flipping sign of y here!!!!!!!!!
                rr, cc = polygon(ylist, xlist, [nc,nc])  # changing order here!!!!
                # lists of rows, colummn
                img_arr[f,rr,cc] = 1  # fill polygon
             
        f+= 1
        
    return img_arr
    
    
# display a series of images (square postage stamps)
def disp_img_arr(img_arr):
    nim = img_arr.shape[0]
    nw = int(np.sqrt(nim)) + 1
    fig,axarr = plt.subplots(nw,nw,figsize=(5,5),sharex = True,sharey=True)
    plt.subplots_adjust(hspace=0,wspace=0)
    axarr_lin = axarr.reshape(-1)
    axarr_lin[0].set_aspect(1.)
    for i in range(nim):
        axarr_lin[i].imshow(img_arr[i,:,:])


        
# make a filename string (useful for generating images for a movie)
def mkfilename(root,d,suffix):
    js = ''
    if (d < 10):
        js = js + '0'
    if (d < 100):
        js = js + '0'
    if (d < 1000):
        js = js + '0'
    js = js + '{:d}'.format(d)
    fs = root + '_' + js + suffix
    return fs


# fill a list with rotations that are near identity (but up to pi/2 away)
# for the octahedron, the smallest rotation of a group element is a rotations by pi/2
# nt should be 6 or 7 and lets you decide how many tilts to do
# we choose chose rotation axes in the xy plane and also choose how far over to rotate
#     on these axes
# calls: scipy transformations rotation routine R.from_rotvec()
def rot_group_octa_nearby(nt):
    # generators for rotation group of octahedron
    rot_list  = []  # to hold all the rotations
    ahat = np.array([0.0,0.0,1.0]) # rotate about z axis
    
    iden = R.from_rotvec(0*ahat)  #identity
    #nt = 7
    for i in range(nt+1):   # now many tilts
        if (i == 0):
            rot_list = rot_list + [iden]
        else:
            tilt = np.pi/3.5/nt *i  # how far over to tilt
            nj = int(np.pi/2*i)
            for j in range(nj):  # how many possible axis orientations with axis in xy plane
                phi = np.pi/2/nj * j  # axis orientations go from 0 to pi/2
                axis = np.array([np.cos(phi),np.sin(phi),0]) # let the axis rotate
                brot = R.from_rotvec(tilt * axis)
                rot_list = rot_list + [brot]  # append rotation to rot_list
                
    print('number of rotations = {:d}'.format(len(rot_list)))
    return rot_list
    
    
# fill a list with rotations that are near identity (but up to phi away)
# for the octahedron, the smallest rotation of a group element is phi_max=pi/2
# for the tetrahedron, the group elements are rotations by 120 degrees which is
#   phi_max = 2pi/3
# nt should be 6 or 7 and lets you decide how many tilts to do
# we choose chose rotation axes in the xy plane and also choose how far over to rotate
# on these axes
# calls: scipy transformations rotation routine: R.from_rotvec()
def rot_group_nearby(nt,phi_max):
    # generators for rotation group of octahedron
    rot_list  = []  # to hold all the rotations
    ahat = np.array([0.0,0.0,1.0]) # rotate about z axis
    
    iden = R.from_rotvec(0*ahat)  #identity
    #nt = 7
    for i in range(nt+1):   # now many tilts
        if (i == 0):
            rot_list = rot_list + [iden]
        else:
            tilt = phi_max/nt/1.8 *i  # how far over to tilt
            nj = int(phi_max*i)
            for j in range(nj):  # how many possible axis orientations with axis in xy plane
                phi = (phi_max/nj) * j  # axis orientations go from 0 to phi_max
                axis = np.array([np.cos(phi),np.sin(phi),0]) # let the axis rotate
                brot = R.from_rotvec(tilt * axis)
                rot_list = rot_list + [brot]  # append rotation to rot_list
                
    print('number of rotations = {:d}'.format(len(rot_list)))
    return rot_list
    
    
# extend the nearby rotations by rotating about z with nphi possibilities spanning from 0 to 2pi
# nphi is number of rotations (radians)
def rot_group_extend(rot_list,nphi):
    ahat = np.array([0.0,0.0,1.0]) # rotate about z axis
    dphi = 2*np.pi/nphi
    new_rot_list = []
    for rot in rot_list:
        for i in range(nphi):
            phi = dphi*i
            rot_phi = R.from_rotvec(phi*ahat)*rot
            new_rot_list = new_rot_list + [rot_phi]
            
    print('number of rotations = {:d}'.format(len(new_rot_list)))
    return new_rot_list
        

# create a set of rotations near a rotation rot0
# generate them so they span a maximum angular distance `angle' from rot0
# generate a grid of them
# inputs:
#    rot: this is a rotation. We want to return rotations close to this rotation
#    angle: float.  maximum angle of variations in rotation
#    nl:  integer that sets grid size.  if nl =5 then a 5x5x5 grid is done and 125 rotation returned
# returns:
#   a list of rotations near rot0
def rot_set_near(rot0,angle,nl):
    rot_list = []  # for the rotation list
    lin_angles = np.linspace(-angle,angle,nl) # has nl elements
    xhat = np.array([1.,0.,0.])
    yhat = np.array([0.,1.,0.])
    zhat = np.array([0.,0.,1.])
    iden = R.from_rotvec(0*xhat)
    npieces = 10;
    for i in range(nl):
        alpha_x = lin_angles[i]
        for j in range(nl):
            alpha_y = lin_angles[j]
            for k in range(nl):
                alpha_z = lin_angles[k]
                rot = iden
                for n in range(npieces):  # build up from infinitesimal rotations
                    rot = R.from_rotvec(xhat*alpha_x/npieces)*\
                          R.from_rotvec(yhat*alpha_y/npieces)*\
                          R.from_rotvec(zhat*alpha_z/npieces)*rot
                # this iteration might use Slerp (spherical interp)? scipy.spatial.transform.Slerp
                    
                rot_list = rot_list + [rot*rot0]
    print('number of rotations = {:d}'.format(len(rot_list)))
    return rot_list
    
    
# run skimages routine match_template on image fimage with a set of image templates
# that are in template_arr
# return the peak values of the cross correlation
# return and the rows and columns of the best value
# calls:  match_template (from skimage)
def matches(fimage, template_arr):
    nt = template_arr.shape[0]  # number of templates
    peak_list = np.zeros(nt)  # value of each peak
    rlist = np.zeros(nt)  # row for each peak
    clist = np.zeros(nt)  # column for each peak
    for k in range(nt):
        template = np.squeeze(template_arr[k,:,:])
        result = match_template(fimage,template,pad_input=True)
        # with pad_input=True, result has same dimensions as trial_image
        ij = np.unravel_index(np.argmax(result), result.shape)
        r, c = ij[::-1]
        #print(r,c,result[ij])  # these are positions within result and amplitude of peak
        rlist[k] = r  # is actually horizontal pixel position  (is col number from left)
        clist[k] = c  # is actually vertical pixel position (is row number from top)
        peak_list[k] = result[ij]
    return rlist,clist,peak_list


# find the best matching rotation and return the rotation, the position and the best template
# inputs:
#   edges -  is a canny edge detected image
#   template_arr - is the array of template images
#   rot_list - list of rotations (that previously made the templates)
# calls: above routine matches()
def find_best_match(edges,template_arr,rot_list):
    rlist,clist,peak_list=matches(edges, template_arr)  # find best match
    k= np.argmax(peak_list)  # the rotation that wins
    #print('find_best_match',k)
    template = np.squeeze(template_arr[k,:,:])
    rot = rot_list[k]
    r = rlist[k]; c = clist[k]
    return rot,k,r,c,template

# display the result of the search for the best matching template
# inputs:
#   subim - the original image
#   edges -  is a canny edge detected result on subimage
#   template - is the best template found
#   rot   - the best rotation
#   scale_fac - scaling factor for model size
#   r,c - row, column indices of best template shift  r from left, c from top
#   vertices, faces - the octahedron model
# calls: plt_edges_s_ax(), find_visible_edges_poly()
def disp_best_match(subim,edges,r,c,template,rot,scale_fac,vertices,faces):
    fig,axarr = plt.subplots(1,3,sharex=True,sharey=True,dpi=100)
    plt.subplots_adjust(hspace=0,wspace=0)
    nh = int(template.shape[1]/2)
    extent = (r-nh,r+nh,c+nh,c-nh)
    axarr[0].imshow(template,extent=extent)
    axarr[0].plot(r,c,'ro')
    axarr[1].plot(r,c,'ro')
    axarr[1].imshow(edges)
    axarr[2].imshow(subim)
    new_vertices = rot.apply(vertices)
    edge_list = find_visible_edges_poly(new_vertices,faces)
    plt_edges_s_ax(edge_list,new_vertices,axarr[1],scale_fac,r,c,'red',2,1)



# find a subimage near max of the image, display result
# inputs:
#   frames: pims sequence of images, not rgb, single color!
#   f0   : frame number
#   dw   : half width of subimage
#   canny_sigma, canny_low, canny_high :  parameters for canny edge detection
# returns:
#   r0,c0  : upper left corner location
#   edges : made with canny edge detection (of subimage)
#   subim : extracted subimage
#   ax :  ax of plot
# calls:
#    uses feature.canny() from skimage

def get_canny_edges(frames,f0,dw,canny_sigma,canny_low,canny_high):
    #img = np.array(gaussian(frames[f0],sigma=0))
    img = np.array(frames[f0])
    i0,j0 = np.unravel_index(img.argmax(), img.shape) # brightest spot
    print('approximate center i0 = {:d} j0={:d}'.format(i0,j0))  # rows cols of brightest spot
    
    # make routines robust to window falling off side of image
    itop = i0-dw; ibottom=i0+dw;
    jleft = j0-dw; jright=j0+dw;
    if (i0-dw <0):
        itop = 0; ibottom = 2*dw;
    if (j0-dw <0):
        jleft = 0; jright=2*dw;
    if (i0 + dw >= img.shape[0]):
        itop = img.shape[0] -1 - 2*dw; ibottom = img.shape[0]-1 ;
    if (j0 + dw >= img.shape[1]):
        jleft = img.shape[1] -1 - 2*dw; jright = img.shape[1]-1 ;
    
    c0 = itop;
    r0 = jleft
    subim = img[itop:ibottom,jleft:jright]  #extract subimage
    
    fig,axarr = plt.subplots(1,2,figsize=(5,3),sharex=True,sharey=True)
    plt.subplots_adjust(hspace=0,wspace=0)
    axarr[0].imshow(subim)
    edges = feature.canny(subim, sigma=canny_sigma,low_threshold=canny_low,high_threshold=canny_high)
    axarr[1].imshow(edges)
    #print(np.median(img)) # you want the low threshold near but below the median?
    return r0,c0,edges,subim,axarr

# similar to above routine except that center i0,j0 are specified
# only display result if xdisp==1
# inputs:
#   frames: pims sequence of images, single color!!!!
#   f0   : frame number
#   dw   : half width of subimage
#   canny_sigma, canny_low, canny_high :  parameters for canny edge detection
#   i0, j0: location of center of subimage within frame f0
# returns:
#   r0,c0  : upper left corner location r0 from left, c0 from top
#   edges : made with canny (of subimage)
#   subim : extracted subimage
# uses feature.canny() from skimage
def get_canny_edges_rc(frames,f0,dw,canny_sigma,canny_low,canny_high,i0,j0,xdisp):
    img = np.squeeze(np.array(frames[f0]))
    print('get_edges_rc:',i0-dw,i0+dw,j0-dw,j0+dw)
    
    itop = i0-dw; ibottom=i0+dw;
    jleft = j0-dw; jright=j0+dw;
    if (i0-dw <0):
        itop = 0; ibottom = 2*dw;
    if (j0-dw <0):
        jleft = 0; jright=2*dw;
    if (i0 + dw >= img.shape[0]):
        itop = img.shape[0] -1 - 2*dw; ibottom = img.shape[0]-1;
    if (j0 + dw >= img.shape[1]):
        jleft = img.shape[1] -1 - 2*dw; jright = img.shape[1]-1;
        
    c0 = itop;
    r0 = jleft
    subim = img[itop:ibottom,jleft:jright]  #extract subimage
    
    edges = feature.canny(subim, sigma=canny_sigma,low_threshold=canny_low,high_threshold=canny_high)
    if (xdisp==1):
        fig,axarr = plt.subplots(1,2,figsize=(5,3),sharex=True,sharey=True)
        plt.subplots_adjust(hspace=0,wspace=0)
        axarr[0].imshow(subim)
        axarr[1].imshow(edges)
    #print(np.median(img)) # you want the low threshold near but below the median?
    return r0,c0,edges,subim
    

# find rotations and position of dice in a sequence of images
# inputs:
#   frames: pims list of frames
#   start_rot:   rotation found for frame with index f0
#   rstart, cstart:  position of center of mass in  frame f0 (rows,cols r from left, c from top) in pixels
#   dw: size of subimages used for template matching
#   canny_sigma, canny_low, canny_high:  parameters for canny edge detection
#   f0:  first frame
#   df:  spacing in frames for sequence
#   nf:  number of frames to do
#   tem_scale_fac:  scaling factor so can get model the right size
#   angle:  largest angle rotation to consider for consecutive frames
#   nl: grid length for rotation grid
#   vertices, faces: octahedron model
#
# returns
#   r_cen_list, c_cen_list:  list of positions of center of mass (in pixels, r from left, c from top)
#   f0_list: frame number list
#   frame_rot_list: list of rotations found
# calls:
#   big_bin_rot_list3
#   get_edges_rc
#   find_best_match
#   get_canny_edges_rc
def do_sequence(frames,start_rot,rstart,cstart,dw,canny_sigma,canny_low,canny_high,\
                f0,df,nf,tem_scale_fac,angle,nl,vertices,faces):

    rot0 = start_rot  # initial rotation for f0 frame
    i0 = int(cstart)  # center guesses , c is pixel from left, r is pixel from top
    j0 = int(rstart)  # i0,j0 used to center subimage
    
    r_cen_list = []  # for storing results
    c_cen_list = []
    f0_list = []
    frame_rot_list = []
    
    for f in range(f0,f0+df*nf,df):  #loop over image frames
        print('doing frame {:d}'.format(f))
        # print(f,i0,j0)
        # generate a finer grid of rotations near initial rotation, can't we just multiply a predone list?
        nrot_list = rot_set_near(rot0,angle,nl)
        # disp_rot_list(nrot_list,vertices,faces)
        # use that nrot_list to make a set of templates
        ntemplate_arr=big_bin_rot_list3(nrot_list,vertices,faces,tem_scale_fac,3)  # make templates
        # make the canny edge image in a subimage, centered at i0,j0, with half width dw
        r0,c0,edges,subim=get_canny_edges_rc(frames,f,dw,canny_sigma,\
                    canny_low,canny_high,i0,j0,0) # 0 means don't display
        if (r0==-1):  # error catch (what if desired subimage extends past edges of image?)
            return r_cen_list,c_cen_list,f0_list,frame_rot_list # return early with all information
    
        # now find the best rotation from the finer grid by template matching
        best_rot,k,r,c,template= find_best_match(edges,ntemplate_arr,nrot_list)
        disp_best_match(subim,edges,r,c,template,best_rot,tem_scale_fac,vertices,faces) # display result
        
        r_cen_list = np.append(r_cen_list,r+r0) # store center, r is from left, c from top
        c_cen_list = np.append(c_cen_list,c+c0)
        f0_list = np.append(f0_list,f)  # store index of frame
        frame_rot_list = frame_rot_list + [best_rot]  # store best rotation
        
        rot0 = best_rot # reset so that next image starts with this one
        i0= int(c+c0)   # reset center so that next subimage starts shifted over
        j0= int(r+r0)
        
    f0_list = np.array(f0_list, dtype=int)  # make sure is ints
        
    # return positions and best rotations
    return r_cen_list,c_cen_list,f0_list,frame_rot_list
    

# compute the spin vectors for a sequence of rotations
# inputs:
#   rot_list: a sequence of rotations for a sequence of frames
#   f0_list: the frame list
#   fps:  frames per second , used to return spin vector in units of rad/s
def compute_spin_vec(rot_list,f0_list,fps):
    nr = len(rot_list)
    df = f0_list[1] - f0_list[0]
    dt = df/fps
    omega_arr = np.zeros((nr,3))
    for i in range(nr-1):
        q0 = rot_list[i]
        q1 = rot_list[i+1]
        trans = q1*q0.inv()
        omega = trans.as_rotvec()/dt
        omega_arr[i,:] = omega
    return omega_arr
    
    
# plot a sequence of projected edges on a background frame
# calls: plt_edges_s_ax(), find_visible_edges_poly()
# inputs:
#   r_cen_list, c_cen_list:  coordinates of center of mass in pixels
#   f0_list: index of frames used
#   frame_rot_list:  list of rotations for each frame done
#   scale fac: scaling factor for model (to convert model to pixels)
#   frame: # background rgb frame for image
#   ddf: is spacing of frames (within the list of frames) we want for our sequence
#   vertices, faces: those of model
#   bbox:  in pixels what to show
#   pixscale:   pix/mm for scale bar
#   ecolor: color of edges plotted
#   lw: line width for edges
#   ofile: filename for png figure
def plt_sequence(r_cen_list,c_cen_list,f0_list,frame_rot_list,scale_fac,\
                 frame,ddf,vertices,faces,bbox,pixscale,ecolor,lw,ofile):
    fig,ax = plt.subplots(1,1,figsize=(4,6),dpi=250)
    #ax.axes.xaxis.set_ticklabels([])
    #ax.axes.yaxis.set_ticklabels([])
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ii1 = bbox[0]
    ii2 = bbox[1]
    jj1 = bbox[2]
    jj2 = bbox[3]
    ax.imshow(frame[ii1:ii2,jj1:jj2])# [0:400,200:800])
    ax.plot(r_cen_list-jj1,c_cen_list-ii1,'-',color='white',lw=1)  # plot center of mass
    for i in range(0,len(r_cen_list),ddf):  # loop over rotations
        rot = frame_rot_list[i]
        new_vertices = rot.apply(vertices)  # generate rotation
        edge_list = find_visible_edges_poly(new_vertices,faces)  # find edges
        r = r_cen_list[i]-jj1
        c = c_cen_list[i]-ii1
        plt_edges_s_ax(edge_list,new_vertices,ax,scale_fac,r,c,ecolor,lw,1) # plot edges
        
    # pixscale is mm/pix
    barlen = 1.0/pixscale*10 # for 1 cm scalebar
    #print(barlen)
    ax.plot([jj2-jj1-barlen-30,jj2-jj1-30],[20,20],'-',lw=2,color='yellow') # 1cm scalebar
    
    if (len(ofile)>3):  # save figure
        fig.savefig(ofile,dpi=200)
        
#test
#frame = framesrgb[50]
#ddf = 1; bbox = [0,400,200,1250]
#plt_sequence(r_cen_list,c_cen_list,f0_list,frame_rot_list,tem_scale_fac,\
#             frame,ddf,vertices,faces,bbox,pixscale,"")
    


# a color combo red+blue
# this routine also reduces dimension and size of frames
# i0,i1,j0,j1 gives region of image frame we want
# converts arrays to floats
# returned is a framelist of single frames (not rgb)
# a,b,c are weights for each color r,g,b
# a,b,c could be negative
@pipeline
def color_rgb(frame,i0,i1,j0,j1,a,b,c):
    red  = np.array(frame[i0:i1,j0:j1,0])
    redf = red.astype('float')
    green  = np.array(frame[i0:i1,j0:j1,1])
    greenf = green.astype('float')
    blue  = np.array(frame[i0:i1,j0:j1,2])
    bluef = blue.astype('float')
    y = a*redf +  b*greenf + c*bluef  # weighted by color
    i = frame.frame_no
    z= Frame(y)
    z.frame_no = i
    # restore frame number information which is needed by trackpy
    return z
