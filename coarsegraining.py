##### Lena Dogra, February 2024
##### Coarse graining PALM data and the turbulent advection terms, saving as .nc file

import numpy as np
import pandas as pd
import xarray as xr
import dask as dd
import os, sys
import glob
dd.config.set(**{'array.slicing.split_large_chunks': False, "dtype": "float32", "zlib": True, "complevel": 4})#### need to set False or the coarse graining will crash


factor = 200
factor_z = 10


### helper function: interpolate on the grid of the state variables
def make_same_coords(ds, grid = 'theta',**interp_kwargs):
    
    if grid == 'theta':
        ### interpolate on the grid of the state variables
        u1 = ds.u.interp(xu = ds.x.values,**interp_kwargs)
        v1 = ds.v.interp(yv = ds.y.values,**interp_kwargs)
        w1 = ds.w.interp(zw_3d = ds.zu_3d.values,**interp_kwargs)

        u1 = u1.rename({'xu':'x'})
        v1 = v1.rename({'yv':'y'})
        w1 = w1.rename({'zw_3d':'zu_3d'})
        
        ds['u1'] = u1
        ds['v1'] = v1
        ds['w1'] = w1
    
    elif grid == 'velocities':
        ### interpolate on the grid of the state variables
        thu = ds.theta.interp(x = ds.xu.values,**interp_kwargs)
        thv = ds.theta.interp(y = ds.yv.values,**interp_kwargs)
        thw = ds.theta.interp(zu_3d = ds.zw_3d.values,**interp_kwargs)

        thu = thu.rename({'x':'xu'})
        thv = thv.rename({'y':'yv'})
        thw = thw.rename({'zu_3d':'zw_3d'})
        
        ds['thu'] = thu
        ds['thv'] = thv
        ds['thw'] = thw

    return ds

# helper to calculate the flux
def make_SGS(ds):
    ds["theta_w_SGS"] = ds["theta_w"] - ds["theta_w_cg"]  + ds["diff_theta_w_cg"]
    ds["theta_v_SGS"] = ds["theta_v"] - ds["theta_v_cg"] + ds["diff_theta_v_cg"]
    ds["theta_u_SGS"] = ds["theta_u"] - ds["theta_u_cg"] + ds["diff_theta_u_cg"]

    ds["u_w_SGS"] = ds["uw_cg"]*ds["w_cg"] - ds["u_w_cg"]+ds["diff_u_w_cg"]
    ds["v_w_SGS"] = ds["vw_cg"]*ds["w_cg"] - ds["v_w_cg"]+ds["diff_v_w_cg"]

    ds["v_u_SGS"] = ds["vu_cg"]*ds["u_cg"] - ds["v_u_cg"]+ds["diff_v_u_cg"]
    ds["w_u_SGS"] = ds["wu_cg"]*ds["u_cg"] - ds["w_u_cg"]+ds["diff_w_u_cg"]

    ds["u_v_SGS"] = ds["uv_cg"]*ds["v_cg"] - ds["u_v_cg"]+ds["diff_u_v_cg"]
    ds["w_v_SGS"] = ds["wv_cg"]*ds["v_cg"] - ds["w_v_cg"]+ds["diff_w_v_cg"]
    return ds

for time_index in np.arange(1,277):
    print(time_index)

    myfile ='/scratch/b/b309252/palm/JOBS/qml2d/OUTPUT/qml2d_3d_0000%03d.nc'%time_index
    
    ds_3d_all = xr.open_mfdataset(myfile, combine = 'by_coords', chunks = 10000, format='netCDF4')

    #format time from timedelta to normal values in seconds
    ds_3d_all['time']  =  np.round(ds_3d_all['time'].astype('timedelta64[ns]') / np.timedelta64(1, 's'), 1)

    #select height range - To Do: select seperately for time steps!
    ds_3d_all = ds_3d_all.sel(zu_3d = slice(1,1500), zw_3d = slice(1,1500))
    interp_kwargs = {"kwargs":{"fill_value": "extrapolate"}}
    
    ds_3d = ds_3d_all.interpolate_na(**interp_kwargs)
    
    #calculate advective fluxes of velocities
    uw = ds_3d.u.interp(xu = ds_3d.x.values).interp(zu_3d = ds_3d.zw_3d.values,**interp_kwargs).rename({'xu':'x', 'zu_3d':'zw_3d'})
    vw = ds_3d.v.interp(yv = ds_3d.y.values).interp(zu_3d = ds_3d.zw_3d.values,**interp_kwargs).rename({'yv':'y', 'zu_3d':'zw_3d'})
    ds_3d['uw'] = uw
    ds_3d['vw'] = vw
    u_w = xr.apply_ufunc(np.multiply,ds_3d.w,ds_3d.uw, dask = 'parallelized')
    v_w = xr.apply_ufunc(np.multiply,ds_3d.w,ds_3d.vw, dask = 'parallelized')

    vu = ds_3d.v.interp(yv = ds_3d.y.values, x = ds_3d.xu.values,**interp_kwargs).rename({'yv':'y','x':'xu'})
    wu = ds_3d.w.interp(zw_3d = ds_3d.zu_3d.values, x = ds_3d.xu.values,**interp_kwargs).rename({'zw_3d':'zu_3d','x':'xu'})
    ds_3d['vu'] = vu
    ds_3d['wu'] = wu
    v_u = xr.apply_ufunc(np.multiply,ds_3d.u,ds_3d.vu, dask = 'parallelized')
    w_u = xr.apply_ufunc(np.multiply,ds_3d.u,ds_3d.wu, dask = 'parallelized')

    uv = ds_3d.u.interp(y = ds_3d.yv.values, xu = ds_3d.x.values,**interp_kwargs).rename({'y':'yv','xu':'x'})
    wv = ds_3d.w.interp(zw_3d = ds_3d.zu_3d.values, y = ds_3d.yv.values,**interp_kwargs).rename({'zw_3d':'zu_3d','y':'yv'})
    ds_3d['uv'] = uv
    ds_3d['wv'] = wv
    u_v = xr.apply_ufunc(np.multiply,ds_3d.v,ds_3d.uv, dask = 'parallelized')
    w_v = xr.apply_ufunc(np.multiply,ds_3d.v,ds_3d.wv, dask = 'parallelized')

    ds_3d['v_w']= v_w
    ds_3d['u_w']= u_w
    ds_3d['v_u']= v_u
    ds_3d['w_u']= w_u
    ds_3d['u_v']= u_v
    ds_3d['w_v']= w_v

    # calculate diffusive fluxes of velocities
    diff_u_w = xr.apply_ufunc(np.multiply,ds_3d.km.interp(zu_3d = ds_3d.zw_3d.values,**interp_kwargs).rename({'zu_3d':'zw_3d'}),ds_3d.u.interp(zu_3d = ds_3d.zw_3d.values,xu = ds_3d.x.values,**interp_kwargs).rename({'xu':'x','zu_3d':'zw_3d'}).differentiate('zw_3d')+ds_3d.w.differentiate('x'), dask = 'parallelized')
    diff_v_w = xr.apply_ufunc(np.multiply,ds_3d.km.interp(zu_3d = ds_3d.zw_3d.values,**interp_kwargs).rename({'zu_3d':'zw_3d'}),ds_3d.v.interp(zu_3d = ds_3d.zw_3d.values,yv = ds_3d.y.values,**interp_kwargs).rename({'yv':'y','zu_3d':'zw_3d'}).differentiate('zw_3d')+ds_3d.w.differentiate('y'), dask = 'parallelized')
    diff_v_u = xr.apply_ufunc(np.multiply,ds_3d.km.interp(x = ds_3d.xu.values,**interp_kwargs).rename({'x':'xu'}),ds_3d.v.interp(yv = ds_3d.y.values,x = ds_3d.xu.values,**interp_kwargs).rename({'yv':'y','x':'xu'}).differentiate('xu')+ds_3d.u.differentiate('y'), dask = 'parallelized')
    diff_w_u = xr.apply_ufunc(np.multiply,ds_3d.km.interp(x = ds_3d.xu.values,**interp_kwargs).rename({'x':'xu'}),ds_3d.w.interp(zw_3d = ds_3d.zu_3d.values,x = ds_3d.xu.values,**interp_kwargs).rename({'zw_3d':'zu_3d', 'x':'xu'}).differentiate('xu') + ds_3d.u.differentiate('zu_3d'), dask = 'parallelized')
    diff_u_v = xr.apply_ufunc(np.multiply,ds_3d.km.interp(y = ds_3d.yv.values,**interp_kwargs).rename({'y':'yv'}),ds_3d.u.interp(y = ds_3d.yv.values, xu = ds_3d.x.values,**interp_kwargs).rename({'y':'yv','xu':'x'}).differentiate('yv') +ds_3d.v.differentiate('x'), dask = 'parallelized')
    diff_w_v = xr.apply_ufunc(np.multiply,ds_3d.km.interp(y = ds_3d.yv.values,**interp_kwargs).rename({'y':'yv'}),ds_3d.w.interp(zw_3d = ds_3d.zu_3d.values,y=ds_3d.yv.values,**interp_kwargs).rename({'zw_3d':'zu_3d', 'y':'yv'})+ ds_3d.v.differentiate('zu_3d'), dask = 'parallelized')

    ds_3d['diff_u_w']= diff_v_w
    ds_3d['diff_v_w']= diff_u_w
    ds_3d['diff_v_u']= diff_v_u
    ds_3d['diff_w_u']= diff_w_u
    ds_3d['diff_w_v']= diff_w_v
    ds_3d['diff_u_v']= diff_u_v

    ps.make_same_coords(ds_3d, grid = 'velocities',**interp_kwargs)
    ds_3d['theta_w']= ds_3d.w*ds_3d.thw
    ds_3d['theta_u']= ds_3d.u*ds_3d.thu
    ds_3d['theta_v']= ds_3d.v*ds_3d.thv

    # calculate diffusive fluxes of theta
    diff_theta_w = xr.apply_ufunc(np.multiply,ds_3d.kh.interp(zu_3d = ds_3d.zw_3d.values,**interp_kwargs).rename({'zu_3d':'zw_3d'}),ds_3d.theta.interp(zu_3d = ds_3d.zw_3d.values,**interp_kwargs).rename({'zu_3d':'zw_3d'}).differentiate('zw_3d'), dask = 'parallelized')
    diff_theta_u = xr.apply_ufunc(np.multiply,ds_3d.kh.interp(x = ds_3d.xu.values,**interp_kwargs).rename({'x':'xu'}),ds_3d.theta.interp(x = ds_3d.xu.values,**interp_kwargs).rename({'x':'xu'}).differentiate( 'xu'), dask = 'parallelized')
    diff_theta_v = xr.apply_ufunc(np.multiply,ds_3d.kh.interp(y = ds_3d.yv.values,**interp_kwargs).rename({'y':'yv'}),ds_3d.theta.interp(y = ds_3d.yv.values,**interp_kwargs).rename({'y':'yv'}).differentiate('yv'), dask = 'parallelized')

    ds_3d['diff_theta_w']= diff_theta_w
    ds_3d['diff_theta_u']= diff_theta_u
    ds_3d['diff_theta_v']= diff_theta_v
    
    #### coarse grain state variables

    ds_cg = ds_3d[['p','theta','ri', 'ti']].coarsen(dim={"x": factor, "y": factor,"zu_3d":factor_z}, boundary="trim").mean()

    ## coarse grain velocities using sides of cells
    ds_cg['u_cg'] = ds_3d.u[:,:,:,::factor].coarsen(dim={"y": factor,"zu_3d":factor_z}, boundary="trim").mean()
    ds_cg['vu_cg'] = ds_3d.vu[:,:,:,::factor].coarsen(dim={"y": factor,"zu_3d":factor_z}, boundary="trim").mean()
    ds_cg['wu_cg'] = ds_3d.wu[:,:,:,::factor].coarsen(dim={"y": factor,"zu_3d":factor_z}, boundary="trim").mean()

    ds_cg['v_cg'] = ds_3d.v[:,:,::factor,:].coarsen(dim={"x": factor, "zu_3d":factor_z}, boundary="trim").mean()
    ds_cg['uv_cg'] = ds_3d.uv[:,:,::factor,:].coarsen(dim={"x": factor, "zu_3d":factor_z}, boundary="trim").mean()
    ds_cg['wv_cg'] = ds_3d.wv[:,:,::factor,:].coarsen(dim={"x": factor, "zu_3d":factor_z}, boundary="trim").mean()

    ds_cg['w_cg'] = ds_3d.w[:,::factor_z,:,:].coarsen(dim={"y": factor,"x":factor}, boundary="trim").mean()
    ds_cg['uw_cg'] = ds_3d.uw[:,::factor_z,:,:].coarsen(dim={"y": factor,"x":factor}, boundary="trim").mean()
    ds_cg['vw_cg'] = ds_3d.vw[:,::factor_z,:,:].coarsen(dim={"y": factor,"x":factor}, boundary="trim").mean()

    ### coarse grain advective and diffusive fluxes using sides of cells
    cg_v_w = ds_3d.v_w[:,::factor_z,:,:].coarsen(dim={"x": factor,"y":factor}, boundary="trim").mean()
    cg_u_w = ds_3d.u_w[:,::factor_z,:,:].coarsen(dim={"y": factor,"x":factor}, boundary="trim").mean()

    cg_v_u = ds_3d.v_u[:,:,:,::factor].coarsen(dim={"y": factor,"zu_3d":factor_z}, boundary="trim").mean()
    cg_w_u = ds_3d.w_u[:,:,:,::factor].coarsen(dim={"y": factor,"zu_3d":factor_z}, boundary="trim").mean()

    cg_u_v = ds_3d.u_v[:,:,::factor,:].coarsen(dim={"zu_3d": factor_z,"x":factor}, boundary="trim").mean()
    cg_w_v = ds_3d.w_v[:,:,::factor,:].coarsen(dim={"zu_3d": factor_z,"x":factor}, boundary="trim").mean()

    ds_cg['v_w_cg'] = cg_v_w
    ds_cg['u_w_cg'] = cg_u_w
    ds_cg['v_u_cg'] = cg_v_u
    ds_cg['w_u_cg'] = cg_w_u
    ds_cg['u_v_cg'] = cg_u_v
    ds_cg['w_v_cg'] = cg_w_v

    cg_diff_v_w = ds_3d.diff_v_w[:,::factor_z,:,:].coarsen(dim={"x": factor,"y":factor}, boundary="trim").mean()
    cg_diff_u_w = ds_3d.diff_u_w[:,::factor_z,:,:].coarsen(dim={"y": factor,"x":factor}, boundary="trim").mean()

    cg_diff_v_u = ds_3d.diff_v_u[:,:,:,::factor].coarsen(dim={"y": factor,"zu_3d":factor_z}, boundary="trim").mean()
    cg_diff_w_u = ds_3d.diff_w_u[:,:,:,::factor].coarsen(dim={"y": factor,"zu_3d":factor_z}, boundary="trim").mean()

    cg_diff_u_v = ds_3d.diff_u_v[:,:,::factor,:].coarsen(dim={"zu_3d": factor_z,"x":factor}, boundary="trim").mean()
    cg_diff_w_v = ds_3d.diff_w_v[:,:,::factor,:].coarsen(dim={"zu_3d": factor_z,"x":factor}, boundary="trim").mean()

    ds_cg['diff_v_w_cg'] = cg_diff_v_w
    ds_cg['diff_u_w_cg'] = cg_diff_u_w
    ds_cg['diff_v_u_cg'] = cg_diff_v_u
    ds_cg['diff_w_u_cg'] = cg_diff_w_u
    ds_cg['diff_u_v_cg'] = cg_diff_u_v
    ds_cg['diff_w_v_cg'] = cg_diff_w_v

    cg_diff_theta_u = ds_3d.diff_theta_u[:,:,:,::factor].coarsen(dim={"y": factor,"zu_3d":factor_z}, boundary="trim").mean()
    cg_diff_theta_v = ds_3d.diff_theta_v[:,:,::factor,:].coarsen(dim={"zu_3d": factor_z,"x":factor}, boundary="trim").mean()
    cg_diff_theta_w = ds_3d.diff_theta_w[:,::factor_z,:,:].coarsen(dim={"y": factor,"x":factor}, boundary="trim").mean()

    cg_theta_u = ds_3d.theta_u[:,:,:,::factor].coarsen(dim={"y": factor,"zu_3d":factor_z}, boundary="trim").mean()
    cg_theta_v = ds_3d.theta_v[:,:,::factor,:].coarsen(dim={"zu_3d": factor_z,"x":factor}, boundary="trim").mean()
    cg_theta_w = ds_3d.theta_w[:,::factor_z,:,:].coarsen(dim={"y": factor,"x":factor}, boundary="trim").mean()

    ds_cg['theta_u_cg'] = cg_theta_u
    ds_cg['theta_v_cg'] = cg_theta_v
    ds_cg['theta_w_cg'] = cg_theta_w
    ds_cg['diff_theta_u_cg'] = cg_diff_theta_u
    ds_cg['diff_theta_v_cg'] = cg_diff_theta_v
    ds_cg['diff_theta_w_cg'] = cg_diff_theta_w


    ps.make_same_coords(ds_cg, grid = 'velocities',**interp_kwargs)
    theta_w_1 = ds_cg.w_cg*ds_cg.thw
    theta_u_1 = ds_cg.u_cg*ds_cg.thu
    theta_v_1 = ds_cg.v_cg*ds_cg.thv

    ds_cg['theta_w']= theta_w_1
    ds_cg['theta_u']= theta_u_1
    ds_cg['theta_v']= theta_v_1

    print('************** Finished coarse graining.')

    make_SGS(ds_cg)

    ds_cg = ds_cg.astype('float32')

    print('************** Finished calculating fluxes.')

    savepath = '/work/bd1179/b309252/qml2_cg/qml2d_0_cg%i_%iz_interpna_000'%(factor,factor_z)+'%03d.nc'%(time_index)
    ds_cg.to_netcdf(savepath)
    print('************** Saved data as %s'%savepath)
