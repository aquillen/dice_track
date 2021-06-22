
from __future__ import division, unicode_literals, print_function  
import numpy as np
import matplotlib.pyplot as plt
import pims  # image reading routines
import os

import pandas as pd
import trackpy as tp
from scipy import ndimage  # for shifting images
from pims import pipeline
from pims import Frame
from scipy.optimize import minimize  # for fitting
#import cv2 as cv2  # for Hough transform circle finding
from scipy.interpolate import griddata
import matplotlib.patches as mpatches
from scipy.signal import medfilt
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter




# rescale to within [0,255]
def normalize255(x):
    xmin = np.min(x)
    xmax = np.max(x)
    y = (x - xmin)/(xmax-xmin)
    y*= 255
    return y


# a color combo red+green+blue
# this routine also reduces dimension and size of frames
# i0,i1,j0,j1 gives region of image frame we want
# converts arrays to floats
@pipeline
def color_rgb(frame,i0,i1,j0,j1,a,b,c):
    red  = np.array(frame[i0:i1,j0:j1,0])
    redf = red.astype('float')
    green  = np.array(frame[i0:i1,j0:j1,1])
    greenf = green.astype('float')
    blue  = np.array(frame[i0:i1,j0:j1,2])
    bluef = blue.astype('float')
    y = a*redf +  b*greenf + c*bluef
    i = frame.frame_no
    z= Frame(y)
    z.frame_no = i  
    # restore frame number information which is needed by trackpy
    return z

# sum two image frames, second one is multiplied by fac
@pipeline
def sumim(frame1,frame2,fac):
    return frame1 + np.array(frame2)*fac

# some a sequence of images from a video
# di is spacing between frames that are used in the sum
# This works for rbg frames  too
def sumseq(processed_video,di):
    sumim0 = processed_video[3]*0
    ni = len(range(0,len(processed_video),di))
    for i in range(0,len(processed_video),di):
        sumim0 = sumim(sumim0,processed_video[i],1.0/ni)
        #print(i)
    return sumim0

# construct a median image from a sequence of images -- we can subtract it later on from all frames
# f0 is first image
# diff is distance between images used for median
def makemedian(f0,diff,videorgb):
    ff = np.array(videorgb[0])
    n = len(videorgb)
    ns = int(n/diff) + 1
    imarr = np.zeros([ns,ff.shape[0],ff.shape[1]])  # storage
    j=0;
    # stack the images into an array structure that ndimage can handle
    for i in range(f0,n,diff):
        ff = np.array(videorgb[i])  
        ffpp = np.reshape(ff, ff.shape + (1,))
        imarr[j,:,:] += ffpp[:,:,0]
        j += 1
    imarr = imarr[0:j,:,:]  # this is the stacked 2d arrays 
    # I shorten it getting rid of zero frames at the ends 

    medianim = np.median(imarr,axis=0)  # make a median image from the stack
    plt.figure()
    plt.imshow(medianim)  # show the median image, it should be the background
    return medianim

# automated  subtracting the median image from all images in a video
@pipeline
def sub_median(frame,medianim):
    x= frame - medianim  # subtracts the median image
    #x -= np.median(x)
    return x


# identify particles/spots in a series of frames using trackpy
# f0 is first frame used
# rrad is spot radius
# minmass is brightness of spots used in tracking
# video is processed and shifted and color corrected
# fcat: returned is the pandas data structure of points found
def ifac(video,rrad,minmass,f0):
    #nframes=len(video)-f0  #  how many frames we want
    n=len(video)  #  end of the video 
    fcat = tp.locate(video[f0], rrad, invert=False, minmass=minmass)
    for i in range(f0+1,n):
        f = tp.locate(video[i], rrad, invert=False, minmass=minmass)
        df = [fcat,f]
        fcat = pd.concat(df)  # pandas concatenate
    return fcat     # pandas table returned

# does this bomb if the table has nothing in it?

# track, link and remove stubs (short tracks)
# return track pandas table and list of unique tracks
def track_link_stub(video,rrad,minmass,f0,maxdist,memory,stublength):
    # f0 is start image
    # rrad is radius of peak
    # minmass is brightness of peak
    # maxdist is max distance between frames for linking
    # memory is how many frames trajectory can go missing in linking
    fcat = ifac(video,rrad,minmass,f0) # track points
    # link tracks
    ta = tp.link_df(fcat, maxdist, memory=memory)   
    # filter out any trajectory that is too short
    t1a = tp.filter_stubs(ta, stublength) # 
    print('Before:', ta['particle'].nunique())
    print('After:', t1a['particle'].nunique())
    particle_id_array = t1a['particle'].unique()
    print(particle_id_array )
    return t1a,particle_id_array  


#   return max, min frame number and mean x of all tracks
def len_track(tracks):
    particle_id_array = tracks['particle'].unique()
    for ipart in particle_id_array:
        iii = (tracks['particle'] == ipart)
        fmin = np.min(tracks[iii].frame)
        fmax = np.max(tracks[iii].frame)
        xmean = np.mean(tracks[iii].x)
        print(ipart,fmax-fmin,int(xmean))


#   return max, min frame number of a specific track
def len_track_i(tracks,ipart):
    iii = (tracks['particle'] == ipart)
    fmin = np.min(tracks[iii].frame)
    fmax = np.max(tracks[iii].frame)
    return fmax-fmin

# return max extent of x change in a specific track
def dx_track_i(tracks,ipart):
    iii = (tracks['particle'] == ipart)
    xmin = np.min(tracks[iii].x)
    xmax = np.max(tracks[iii].y)
    return np.abs(xmax-xmin)
        
    
# zoom in to a subregion of a video
@pipeline
def zoom_vid(frame,xx0,xx1,yy0,yy1):
    y= frame[yy0:yy1,xx0:xx1]
    return y

# routines for fitting tracks
#https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
#quaturnion rotation of x,y,z
def qrot(x,y,z,qr,qi,qj,qk):
    #qr = np.sqrt(1.0 - qi*qi - qj*qj - qk*qk)  # assume a unit quaturnion
    s=1  # assume a unit quaturnion
    Rxx = 1.0 - 2*s*(qj*qj + qk*qk)
    Rxy = 2*s*(qi*qj - qk*qr)
    Rxz = 2*s*(qi*qk + qj*qr)
    xnew = Rxx*x + Rxy*y + Rxz*z
    Ryx = 2*s*(qi*qj + qk*qr)
    Ryy = 1.0 - 2*s*(qi*qi + qk*qk)
    Ryz = 2*s*(qj*qk - qi*qr)
    ynew = Ryx*x + Ryy*y + Ryz*z
    Rzx = 2*s*(qi*qk - qj*qr)
    Rzy = 2*s*(qj*qk + qi*qr)
    Rzz = 1.0 - 2*s*(qi*qi + qj*qj)
    znew = Rzx*x + Rzy*y + Rzz*z
    return xnew,ynew,znew

# compute new x,y,z positions for rotation 
# this is for the model
# x0,y0 are positions at time i=0 on image
# theta, phi give rotation axis
# radius is that of marble
# omega_i gives rotation rate
def xyznew(index,x0,y0,theta,phi,omega_i,radius):
    alpha = omega_i*index #rotation angle, assuming that x0,y0 at i=0, no rotation
    ux = np.sin(theta)*np.cos(phi)  # rotation axis
    uy = np.sin(theta)*np.sin(phi)
    uz = np.cos(theta)
    sa2 = np.sin(alpha/2)
    ca2 = np.cos(alpha/2)
    #quaturnion is qr,qi,qj,qk
    qi = sa2*ux
    qj = sa2*uy
    qk = sa2*uz
    qr = ca2
    z2 = radius**2 - x0*x0 - y0*y0
    z0 = np.sqrt(np.fabs(z2)+0.00001)## this must work on vectors
    xnew,ynew,znew=qrot(x0,y0,z0,qr,qi,qj,qk) # quaturnion rotate
    return xnew,ynew,znew
  
# generate a model track
def genmodel(coeffs,ivec):
    x0 = coeffs[0]
    y0 = coeffs[1]
    theta = coeffs[2]
    phi = coeffs[3]
    omega_i = coeffs[4]  # this is rotation rate in units of frames
    radius = coeffs[5]
    xnew,ynew,znew=xyznew(ivec,x0,y0,theta,phi,omega_i,radius)
    return xnew,ynew,znew

# minimize this function when fitting a track
# this gives a comparison of model track to actual one
def funmin(coeffs,xdata,ydata,ivec):
    radius = coeffs[5]
    zdata = np.sqrt(np.fabs(radius**2 - xdata**2  - ydata**2))  # positive!
    xnew,ynew,znew = genmodel(coeffs,ivec)
    rarr2 =np.sum((xdata - xnew)**2 + (ydata-ynew)**2 + (zdata-znew)**2)/len(xdata) 
    # difference between data and model
    rarr = np.sqrt(rarr2)  # rarr2 is chi^2/N
    return rarr  # this should be a measure of scatter from the line
    


# return arrays of [x,y] positions for a single track, 
# also returns frame index list for each tracked point
# made from images with center of mass subbed out
# ipart is the index of the track (or particle)
# track is the table of tracks from trackpy
# returns xy positions of particle track
def xywrap(track,ipart,cx0,cy0):
    iii = (track['particle'] == ipart)
    #print(iii)
    xarr = np.array(track[iii].x) - cx0
    yarr = np.array(track[iii].y) - cy0
    iarr = np.array(track[iii].frame)
    return xarr,yarr,iarr
  
#####################
# fit an entire track with rotation model
# arguments:
#   the track is a pandas table 
#   ipart is the particle number
#   cx0,cy0 is the center of marble
#   diam is diameter of marble in pixels
#   if ydisp==1 then display things
# returns:
#    coefficients of best rotating fit to the track, an array
#      these are [x0,y0,theta,phi,omega_i,radius]
#    sqrt chi^2/N of fit
#    track model positions
#    results contains inverse hessian of fits and other outputs from fits
# The coefficients  of the fit 
#   x0 = coeffs[0]
#   y0 = coeffs[1]
#   theta = coeffs[2]
#   phi = coeffs[3]
#   omega_i = coeffs[4]
# x0,y0 are positions at time i=0 on image
# theta, phi give rotation axis
# radius is that of marble
# omega_i gives rotation rate
def fit_track(track,ipart,cx0,cy0,diam,ydisp):
    xdata,ydata,iarr= xywrap(track,ipart,cx0,cy0)
    x0 = xdata[0]  #from first tracked frame
    y0 = ydata[0]
    ishift = iarr[0]  # this is the frame index shift
    iarr0 = iarr-iarr[0]   # shift frame index!
    phi=-np.pi/2
    omega_i=0.04
    radius = diam/2 # in pixels
    xdiff=5
    ntheta = 5
    dtheta = np.pi/ntheta
    rdist_min = 1e10
    jstor = -1
    # bounds for fit
    bnds = ((x0-xdiff,x0+xdiff),(y0-xdiff,y0+xdiff),\
          (0,np.pi),(-np.pi,np.pi),(None,None),(radius,radius))
    # loop over initial conditions in theta
    for j in range(ntheta):
        theta=dtheta/2 + j*ntheta 
        c0 =[x0,y0,theta,phi,omega_i,radius]  #coefficients for fitting
        # do the fit via scipy,optimize.minimize call
        results = minimize(funmin, c0, args=(xdata,ydata,iarr0),bounds=bnds) 
        #print(results) # is more than coefficients
        rdist = funmin(results.x,xdata,ydata,iarr0)  # return sqrt chi^2/N
        # choose the best fit, and keep track of it
        if (rdist < rdist_min):
            rdist_min = rdist
            jstor = j

    if (ydisp==1):
        print('sqr chi2/N {:.2f}'.format(rdist_min))
    theta=dtheta/2 + jstor*ntheta  # best fit initial theta
    c0 =[x0,y0,theta,phi,omega_i,radius]  #coefficients for fitting
    # redo fit
    results = minimize(funmin, c0, args=(xdata,ydata,iarr0),bounds=bnds )
    coeffs = results.x
    xnew,ynew,znew = genmodel(coeffs,iarr0) # this gives model track
    ii = znew>0  # only front side!
    theta = coeffs[2]
    phi = coeffs[3]
    omega_i = coeffs[4]
    ux = np.sin(theta)*np.cos(phi)
    uy = np.sin(theta)*np.sin(phi)
    uz = np.cos(theta)
    # plot results
    if (ydisp==1):
        plt.figure()
        plt.axes().set_aspect('equal')
        plt.xlim([-radius,radius])
        plt.ylim([-radius,radius])
        plt.plot(xdata,-ydata,'b.')
        plt.plot(xnew[ii],-ynew[ii],'m.')      
        ax = plt.gca()
        circle1 = plt.Circle((0, 0), radius, color='b', fill=False)
        ax.add_artist(circle1)  # show marble
        plt.arrow(0,0,ux*radius,-uy*radius,width=1.1,length_includes_head=True)  # show spin vector
    
    return coeffs,rdist_min,xdata,ydata,xnew,ynew,znew,results


# fit a track but don't allow angles to vary
# theta, phi in radians
def fit_track_noangles(track,ipart,cx0,cy0,diam,ydisp,theta,phi):
    xdata,ydata,iarr= xywrap(track,ipart,cx0,cy0)
    x0 = xdata[0]  #from first tracked frame
    y0 = ydata[0]
    ishift = iarr[0]  # this is the frame index shift
    iarr0 = iarr-iarr[0]   # shift frame index!
    omega_i=0.04
    radius = diam/2 # in pixels
    xdiff=5
    rdist_min = 1e10
    jstor = -1
    # bounds for fit
    bnds = ((x0-xdiff,x0+xdiff),(y0-xdiff,y0+xdiff),\
          (theta,theta),(phi,phi),(None,None),(radius,radius))
   
    c0 =[x0,y0,theta,phi,omega_i,radius]  #coefficients for fitting

    results = minimize(funmin, c0, args=(xdata,ydata,iarr0),bounds=bnds )
    coeffs = results.x
    xnew,ynew,znew = genmodel(coeffs,iarr0) # this gives model track
    ii = znew>0  # only front side!
    theta = coeffs[2]
    phi = coeffs[3]
    omega_i = coeffs[4]
    ux = np.sin(theta)*np.cos(phi)
    uy = np.sin(theta)*np.sin(phi)
    uz = np.cos(theta)
    # plot results
    if (ydisp==1):
        plt.figure()
        plt.axes().set_aspect('equal')
        plt.xlim([-radius,radius])
        plt.ylim([-radius,radius])
        plt.plot(xdata,-ydata,'b.')
        plt.plot(xnew[ii],-ynew[ii],'m.')
        ax = plt.gca()
        circle1 = plt.Circle((0, 0), radius, color='b', fill=False)
        ax.add_artist(circle1)  # show marble
        plt.arrow(0,0,ux*radius,-uy*radius,width=1.1,length_includes_head=True)  # show spin vector
        plt.plot(ux*radius,-uy*radius,'ro')

    return coeffs,rdist_min,xdata,ydata,xnew,ynew,znew,results




# resample cm tracks so covers entire movie 
# working in pixels here
# these gives a cm position for every frame
#  these are for shifting to center of mass frame
def findcm(track,n):  # n should be length of video
    xarr = np.array(track.x)
    yarr = np.array(track.y)
    # shift time to be from impact and put in seconds
    tarr = np.array(track.frame)  

    # where we want to know data
    tarr_even  = np.arange(0,n)
    
    # resample tracks
    xarr_even = griddata(tarr, xarr, tarr_even)
    yarr_even = griddata(tarr, yarr, tarr_even)
    xarr_smo = median_filter(xarr_even,9,mode='nearest')   
    yarr_smo = median_filter(yarr_even,9,mode='nearest')
    #xarr_smo  = xarr_even
    return xarr_smo,yarr_smo  #there are nans in here!


# shift all the images in the video 
# so that the center of mass is no longer moving
# xarr_s,  yarr_s are positions of shifts
@pipeline
def do_shift_video(frame,xarr_s,yarr_s):
    i = frame.frame_no
    shift=[-yarr_s[i],-xarr_s[i]]  # the shift
    x = ndimage.interpolation.shift(frame, shift, output=None, order=3, \
                                  mode='constant', cval=0.0) 
    y= Frame(x)
    y.frame_no = i  
    # restore frame number information which is needed by trackpy
    return y


# return only longer tracks
#mindi is frame number difference
def longer_tracks(tracks,mindi):
    fcat=[]
    particle_id_array = tracks['particle'].unique()
    for ipart in particle_id_array:
        iii = (tracks['particle'] == ipart)
        tab = tracks[iii]
        fmin = np.min(np.array(tab.frame))
        fmax = np.max(np.array(tab.frame))
        #print(fmax-fmin)
        if (fmax - fmin >= mindi):
            if (len(fcat) ==0):
                fcat = tab
            else:
                df = [fcat,tab]
                fcat = pd.concat(df)  # pandas concatenate
    if (fcat.empty):
       particle_id_array_new = [] 
    else:
       particle_id_array_new = fcat['particle'].unique()
    return fcat,particle_id_array_new


# discard all track points outside the marble
# diam is marble diameter
# cx0,cy0 are coordinates of center in pixels
# tracks is pandas table of linked tracks
def discard_outside_radius(tracks,diam,cx0,cy0):
    rads = np.sqrt((tracks.x - cx0)**2 + (tracks.y - cy0)**2)
    iii = (rads < diam/2)
    newtracks = tracks[iii]
    return newtracks


# print the number of unique tracks
def ntracks(tracks):
    particle_id_array = tracks['particle'].unique()
    print(len(particle_id_array))




# display the color adjusted, median subtracted images
# and find center and diameter in pixels of marble (by eye)
# by addjusting x0,y0 and diam
# seq is image
# diam is diameter of marble in Pixels
# sphere_diameter is the actual marble diameter in mm
# vminfac, vmaxfac:  affect display range, they should be larger than 1
# x0,y0 center of a circle to display on top of marble
# return pixel scale in mm
def display_seq(seq,vminfac,vmaxfac,x0,y0,diam,sphere_diameter):
    vmi = np.min(seq); vma = np.max(seq)
    print('vmin, vmax: {:.1f} {:.1f}'.format(vmi,vma))
    plt.figure(figsize=(12,4))
    plt.imshow(seq, cmap='gray' ,vmin=vmi/vminfac,vmax=vma/vmaxfac)  # adjust display
    #diam=63; x0=183; y0=52; # find the pixel scale and marble diameter
    #x1=x0+diam; # note diam is marble diameter in pixels!
    xl = x0-diam/2
    xr = x0+diam/2
    plt.plot([xl,xr],[y0,y0],'r-')
    pixscale = sphere_diameter/diam   # mm/pixel here!
    circle1 = plt.Circle( (x0,y0), diam/2, color='k', fill=False,alpha=0.5)
    ax = plt.gca()
    ax.add_artist(circle1)  # show circle
    #ax.add_artist(ellipse1)  # show ellipse
    print('pixscale is {:.3f} mm/pixel'.format(pixscale))
    return pixscale



# generate a model track for a specific particle
# ip is the number giving the order in the coeffs for the x0,y0 of that particle
def genmodel_ip(coeffs,ivec,ip):
    theta = coeffs[0]
    phi = coeffs[1]
    omega_i = coeffs[2]  # this is rotation rate in units of frames
    radius = coeffs[3]
    #cx0 = coeffs[4]
    #cy0 = coeffs[5]
    x0 = coeffs[6+2*ip]
    y0 = coeffs[6+2*ip+1]
    xnew,ynew,znew=xyznew(ivec,x0,y0,theta,phi,omega_i,radius)
    return xnew,ynew,znew

# coeffs index
# 0  theta (radians)
# 1 phi  (radians)
# 2 omega  units frames-1?
# 3 radius sphere radius
# 4 cx pix center of marble
# 5 cy pix center of marble
#   6 x0 for zero-th track, w.r.t to center of marble
#   7 y0 for zero-th track
#   8 x0 for first track
#   9 y0 for first track
# etc


# minimize this function when fitting tracks
# this gives a comparison of model tracks to actual ones
def funmin_ip(coeffs,tracks):
    theta = coeffs[0]  #spin orientation
    phi = coeffs[1]
    omega_i = coeffs[2]  # this is rotation rate in units of frames
    radius = coeffs[3]  #marble radius
    cx0 = coeffs[4]  #marble center
    cy0 = coeffs[5]
    particle_id_array = tracks['particle'].unique()
    npa = len(particle_id_array)  # number of particles
    rarr2=0
    ndata = 0
    imin = np.min(np.array(tracks.frame))
    for ip in range(npa):  # loop over particles
        ipart = particle_id_array[ip]  #track index
        iii = (tracks['particle'] == ipart)
        #x0 = coeffs[6+2*ip]  # track starts at t=0
        #y0 = coeffs[6+2*ip+1]
        xdata = np.array(tracks[iii].x) - cx0  # w.r.t to center of marble
        ydata = np.array(tracks[iii].y) - cy0
        ivec  = np.array(tracks[iii].frame) - imin
        zdata = np.sqrt(np.fabs(radius**2 - xdata**2  - ydata**2))  # positive!
        xnew,ynew,znew = genmodel_ip(coeffs,ivec,ip)
        rarr2 +=np.sum((xdata - xnew)**2 + (ydata-ynew)**2 + (zdata-znew)**2)
        ndata += len(xdata)
    return rarr2/ndata  #chi^2

# make an output file name
# d is integer
# return a string that is root + integer + suffix
def mkostring(root,suffix,d):
    dstring = '{:d}'.format(d) 
    if (d<10):
        dstring = '0'+dstring 
    if (d<100):
        dstring = '0'+dstring 
    jstring = root+ dstring + suffix
    print(jstring)
    return jstring


# extract the cm trajectory
# return points and smoothed values for
#     x,z,vx,vz,ax,az  as a function of time on an even grid
# x0 pixels  #impact site!
# y0=pixels  #impact site
# t0_frame =  # impact start frame
# theta_cam_deg = 45  #camera angle in degrees
# fps is frames per second
# pixscale is mm/pixel
# ofile is outfile if you want to save a figure
# medlen is length of median filter window -- length in units of gridded data
# vlen is length for savgol filters for vx,vz
# alen is length for savgol filters for ax,az
# return data points and smoothed vectors

def track_trajectory(x0,y0,t0_frame,track,theta_cam_deg,fps,pixscale,medlen,vlen,alen):
    st = np.sin(theta_cam_deg*np.pi/180) # for correcting aspect ratio
    # shift x,z position to be at impact site, units cm 
    # put x,z into cm frame,  correct z for camera angle
    xcorrfac = 1.0  # if  you want to stretch x for some reason (reflection moves?)
    xarr = np.array( (track.x -x0)*pixscale/10)*xcorrfac 
    yarr = np.array(-(track.y -y0)*pixscale/10)/st
    # shift time to be from impact and put in seconds
    iarr = np.array(track.frame - t0_frame)
    tarr = iarr/fps  #units of seconds

    # where we want to know data
    n = len(xarr)
    nsamp = int(n/1.9)  # data every other frame about
    tarr_even  = np.linspace(np.min(tarr),np.max(tarr),nsamp)  # new time vector
    # resample tracks
    xarr_even = griddata(tarr, xarr, tarr_even)
    yarr_even = griddata(tarr, yarr, tarr_even)
    
    # median filter x,y
    xarr_smo = median_filter(xarr_even,medlen,mode='nearest')    
    yarr_smo = median_filter(yarr_even,medlen,mode='nearest')
    dt = tarr_even[1]-tarr_even[0]  # sampling rate
    # compute vx,vy
    vx = np.gradient(xarr_smo)/dt 
    vx_smo = savgol_filter(xarr_smo, vlen, 2,mode='nearest',deriv=1)/dt # window, polyorder
    vy = np.gradient(yarr_smo)/dt
    vy_smo = savgol_filter(yarr_smo, vlen, 2,mode='nearest',deriv=1)/dt  
    # compute ax,ay
    ax = np.gradient(vx_smo)/dt
    ax_smo = savgol_filter(xarr_smo, alen, 2,mode='nearest',deriv=2)/dt**2
    ay = np.gradient(vy_smo)/dt
    ay_smo = savgol_filter(yarr_smo, alen, 2,mode='nearest',deriv=2)/dt**2

    ii =  (tarr_even>-0.015)& (tarr_even<-0.005) # for measuring impact initial velocity
    v = np.sqrt(vx_smo**2 + vy_smo**2)
    theta_arr = np.arctan2(-vy_smo,vx_smo)
    vinit = np.mean(v[ii])  # initial velocity
    theta_init = np.mean(theta_arr[ii])*180.0/np.pi
    # initial impact angle
    print('vinit = {:.1f} cm/s theta_init = {:.1f}'.format(vinit,theta_init))
    v = np.sqrt(vx**2 + vy**2)
    theta_arr = np.arctan2(-vy,vx)
    vinit = np.mean(v[ii])  # initial velocity
    theta_init = np.mean(theta_arr[ii])*180.0/np.pi
    # initial impact angle
    print('vinit = {:.1f} cm/s theta_init = {:.1f}'.format(vinit,theta_init))
    return tarr,xarr,yarr,tarr_even,xarr_smo,yarr_smo,vx,vx_smo,vy,vy_smo,ax,ax_smo,ay,ay_smo
    

# this is a helper routine to help find where impact starts
# track is tracks of cm trajectory
# t0_frame is a guess for frame of start of impact
# this routine gives you the x, y values in pixels at frame t0_frame
def findx0y0(tracks,t0_frame):
    xarr = np.array(tracks.x)
    yarr = np.array(tracks.y)
    # shift time to be from impact and put in seconds
    tarr = np.array((tracks.frame-t0_frame))  

    # where we want to know data
    tarr_even  = np.linspace(np.min(tarr),np.max(tarr),240) 

    # resample tracks
    xarr_even = griddata(tarr, xarr, tarr_even)
    yarr_even = griddata(tarr, yarr, tarr_even)
    i0 = np.argmin(np.abs(tarr_even))  # index of t0
    print('(x0 ,y0) = ({:.1f}, {:.1f}) pixels'.format(xarr_even[i0],yarr_even[i0])) #print frame and pixel no of impact
    return xarr_even[i0],yarr_even[i0]


# make a nice figure showing tracks of dots on top of marble
# tracks is pandas track table
# video is a video
# iframe is which frame to use of video for grayscale background
# cx0,cy0 are center of mass in pixels
# diam is marble diameter in pixels
# pixscale is pixel scale in mm/pix
# vminfac,vmaxfac are for display of grayscale image
# ofile for saving png
def track_fig_cm(tracks,video,iframe,cx0,cy0,diam,pixscale,vminfac,vmaxfac,ofile):
    fig,ax = plt.subplots(1,1,figsize=(5,4))
    plt.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.15, \
        wspace=0.22, hspace=0.0)
    frame = video[iframe]
    dy=frame.shape[0]*pixscale/10
    dx=frame.shape[1]*pixscale/10
    xleft = dx/2
    ybottom = dy/2
    extent =([-dx/2,dx/2,-dy/2,dy/2])
    vmi = np.min(frame); vma = np.max(frame)
    ax.imshow(frame, cmap='gray' ,vmin=vmi/vminfac,vmax=vma/vmaxfac,extent=extent)
    ax.set_xlabel('(cm)',fontsize=18)
    ax.set_ylabel('(cm)',fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    
    colorlist = ['lightcoral','red','orange','sandybrown','gold','palegreen',\
            'cyan','cornflowerblue','plum','magenta','crimson','gray',\
             'lightcoral','red','orange','sandybrown','gold','palegreen',\
             'cyan','cornflowerblue','plum','magenta','crimson','gray'] 
    ncl = len(colorlist)
    particle_id_array = tracks['particle'].unique()
    j=0
    for ipart in particle_id_array:
        iii = (tracks['particle']==ipart)
        xarr = np.array(tracks[iii].x)
        yarr = np.array(tracks[iii].y)
        plt.plot(xarr*pixscale/10 - xleft,ybottom-yarr*pixscale/10,'.',color=colorlist[j%ncl],\
         markersize=0.5)
        j+=1

    cx = cx0*pixscale/10 - xleft
    cy =  ybottom - cy0*pixscale/10 
    circle1 = plt.Circle((cx, cy), diam/2*pixscale/10, color='w', fill=False)
    ax.add_artist(circle1)  # show marble

    if (len(ofile)>3):
        plt.savefig(ofile)



#####################
# fit a bunch of tracks with rotation model
# don't allow rotation angles to vary
# arguments:
#   tracks is a pandas table with tracks and links
#   cx0,cy0 is the center of marble in frame in pixels
#   diam is diameter of marble in pixels
#   theta,phi rotation angle in radians
#   if ydisp==1 then display fit
#   ss display string for right hand corner of plot
#   ofile is png image name for display figure save
def fit_mtracks_noangles(tracks,cx0,cy0,diam,theta,phi,ydisp,ss,ofile):
    particle_id_array = tracks['particle'].unique()
    npa = len(particle_id_array)  # number of particles
    #print(npa)
    omega_i=0.04  #initial condition
    radius = diam/2 # in pixels
    xdiff= radius
    om_max = 20  #limits on omega
    # bounds for fit
    c0 =np.zeros(6+npa*2) 
    bnds = np.zeros((6+npa*2,2), order = 'C')
    bnds[0,0] = theta;  bnds[0,1] = theta;   #all are fixed
    bnds[1,0] = phi;    bnds[1,1] = phi; 
    bnds[2,0] =-om_max; bnds[2,1] = om_max;  # adjustable!
    bnds[3,0] = radius; bnds[3,1] = radius;
    bnds[4,0] = cx0;    bnds[4,1] = cx0;
    bnds[5,0] = cy0;    bnds[5,1] = cy0;
    #initial guesses for coefficients 
    c0[0] = theta
    c0[1] = phi 
    c0[2] = omega_i
    c0[3] = radius
    c0[4] = cx0
    c0[5] = cy0
    for ip in range(npa):
        j = 6 + 2*ip
        bnds[j,0] = -xdiff  # adjustable x0
        bnds[j,1] =  xdiff
        bnds[j+1,0] = -xdiff  # adjustable y0
        bnds[j+1,1] =  xdiff
        ipart = particle_id_array[ip]  #track index
        iii = (tracks['particle'] == ipart)
        c0[j]   = np.median(tracks[iii].x) -cx0  # initial guess for x0
        c0[j+1] = np.median(tracks[iii].y) -cy0  # initial guess for y0
  
    #print('initial coeffs',c0)
    #print('bounds',bnds)
    results = minimize(funmin_ip, c0, args=(tracks),bounds=bnds )
    coeffs=results.x
    omega_r = coeffs[2] 
    vec = c0*0
    vec[2]=1 # for omega
    de_omega = np.sum(vec*(results.hess_inv(vec))) # crude error estimate
    print('omega_i {:.3f} pm {:.3f}'.format(omega_r,de_omega))
    if (ydisp==1):
        colorlist = ['lightcoral','red','orange','sandybrown','gold','palegreen',\
            'cyan','cornflowerblue','plum','magenta','crimson','gray',\
             'lightcoral','red','orange','sandybrown','gold','palegreen',\
             'cyan','cornflowerblue','plum','magenta','crimson','gray'] 
        ncl = len(colorlist)
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.set_aspect('equal')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ux = np.sin(theta)*np.cos(phi)
        uy = np.sin(theta)*np.sin(phi)
        uz = np.cos(theta)
        ax = plt.gca()
        circle1 = plt.Circle((0, 0), 1, color='b', fill=False)
        ax.add_artist(circle1)  # show marble radius
        ax.arrow(0,0,ux,-uy,width=0.02,length_includes_head=True)  # show spin vector
        ax.text(0.92,0.89,ss,fontsize=18,horizontalalignment='right')
        imin = np.min(tracks.frame)
        for ip in range(npa):
            ipart = particle_id_array[ip]  #track index
            iii = (tracks['particle'] == ipart)
            xdata = np.array(tracks[iii].x) - cx0
            ydata = np.array(tracks[iii].y) - cy0
            ivec  = np.array(tracks[iii].frame)-imin
            ax.plot(xdata/radius,-ydata/radius,'.',color=colorlist[ip%ncl],ms=4)
            xnew,ynew,znew = genmodel_ip(coeffs,ivec,ip)
            ii = (znew >0)  # only show the front side!
            ax.plot(xnew[ii]/radius,-ynew[ii]/radius,'k.',ms=3)
        if (len(ofile)>2):
            plt.savefig(ofile)
        
    return omega_r,de_omega

 
# fit tracks in a pandas table but only using tracks in an interval of time
# diam marble diameter in pixels
# fps frames per second
# t0_frame is that when impact started
# cx0,cy0 center of marble in pixels for the video tracked
# f0 start frame only use points after this frame 
# df sets end frame as f0+df  only use points in interval of these frames
# ntl discard tracks with fewer points in the interval than this 
# theta_deg is camera angle -- there may be a cos/sin error here!
# ofile is image file for track display
def omega_df(tracks,diam,fps,t0_frame,cx0,cy0,f0,df,ntl,theta_deg,phi_deg,ofile):
    tracks_d=discard_outside_radius(tracks,diam,cx0,cy0)  # remove track points outside marble boundary
    iii = (tracks_d.frame >=f0) & (tracks_d.frame < f0+df) # small interval in time
    tracks_df = tracks_d[iii]  # only include tracks that are in this interval
    if (tracks_df.empty):
        return -1000,-1000,-1000,-1000  # badvals
    tracks_dfl,particle_id_array = longer_tracks(tracks_df,ntl)  # get rid of short tracks
    npa = len(particle_id_array)
    if (npa ==0):
        return -1000,-1000,-1000,-1000  # badvals
    print('number of tracks ',npa)
    ydisp=1 #display fit
    phi = phi_deg*np.pi/180. # np.pi/2  # assumed!
    theta = theta_deg*np.pi/180  #based on camera angle of 45 
    tmid = float(f0) + float(df)/2.0 
    tmid -= t0_frame
    tmid /= fps  # time of middle of interval
    dt = float(df)/2/fps  # half interval of time in s
    ss = '{:.3f}'.format(tmid)  # time string
    omega_i,de_om_i=fit_mtracks_noangles(tracks_dfl,cx0,cy0,diam,theta,phi,ydisp,ss,ofile)
    omega = omega_i*fps  #rotation rate in s^-1
    de_omega = de_om_i*fps  #error
    return tmid,dt,omega,de_omega
    

