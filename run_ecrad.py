#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:24:12 2020

@author: walther
"""
import os
import shutil
import numpy as np
import xarray as xr
import subprocess
import time

# import modules.sunpos as sp
from trosat import sunpos as sp

def make_namfile_str(ecrad_path = "/home/walther/Programms/ecrad-1.2.0",
                     optpropfile = "aerosol_cams_ifs_optics.nc",
                     aerosol = True,
                     albedo = True,
                     h2o_scale = 1,
                     co2_scale = 1,
                     o3_scale = 1):
    ecrad_path = os.path.abspath(ecrad_path)
    pf = os.path.join(ecrad_path,'data')
    if aerosol:
        a = 'true'
    else:
        a = 'false'
    if albedo:
        al = '!'
    else:
        al = ''
    file = str("""
               &radiation_driver
                do_parallel              = true,   ! Use OpenMP parallelization?
                nblocksize               = 8,      ! Number of columns to process per thread
                do_save_inputs           = false,  ! Save inputs in "inputs.nc"?
                iverbose                 = 3,
                istartcol                = 0,      ! Use full range of columns by default
                iendcol                  = 0,
                nrepeat                  = 1,
                !
                ! Override Input
                h2o_scaling = {h2o},
                co2_scaling = {co2},
                o3_scaling = {o3},
                {albedo}sw_albedo = 0,
                /
                !
                &radiation
                ! General----------------------------------------------------------------------
                do_sw                   = true,           ! Compute shortwave fluxes?
                do_lw                   = true,           ! Compute longwave fluxes?
                do_sw_direct            = true,           ! Compute direct downward shortwave fluxes?
                do_clear                = false,          ! Compute clear-sky fluxes?
                directory_name          = "{pf}",   ! Location of configuration files
                !
                sw_solver_name          = "Cloudless",    ! Solver
                lw_solver_name          = "Cloudless",    ! Solver
                gas_model_name          = "RRTMG-IFS",    ! Gas Model
                !
                iverbosesetup           = 4,              ! Verbosity in setup
                iverbose                = 1,              ! Verbosity in execution
                do_save_spectral_flux   = true,          ! Save flux profiles in each bands
                do_save_gpoint_flux     = false,          ! Save flux profiles in each g-point
                do_surface_sw_spectral_flux= true,        ! Save surface sw fluxes in each band for subsequent diagnostics
                do_lw_derivatives       = false,          ! Compute derivatives for Hogan and Bozzo (2015) approximate updates
                do_save_radiative_properties= false,      ! Write intermediate nc files for properties sent to solver
                do_canopy_fluxes_sw     = false,          ! Save surface shortwave fluxes in each albedo interval
                do_canopy_fluxes_lw     = false,          ! Save surface longwave fluxes in each emissivity interval
                !
                ! SURFACE ALBEDO and EMISSIVITY ----------------------------------------------
                do_nearest_spectral_sw_albedo  = false,
                sw_albedo_wavelength_bound(1:1) =  0.7e-6,
                i_sw_albedo_index(1:2) 		= 1,2,
                ! emissivity is  a boradband value, therefore input has only one dimension
                do_nearest_spectral_lw_emiss 	= false,
                lw_emiss_wavelength_bound(1:1)	= 8.0e-6,
                i_lw_emiss_index(1:2) 		= 1,1,
                !
                ! AEROSOL PROPERTIES----------------------------------------------------------
                use_aerosols                = {a},
                do_lw_aerosol_scattering    = {a},
                aerosol_optics_override_file_name = "{aerfile}",
                n_aerosol_types             = 11,
                i_aerosol_type_map          = -1, -2, -3, 1, 8, 6, -4, 10, 11 ,11, -5,
                /
                """)

    file=file.format(pf=pf,
                     aerfile=optpropfile,
                     a=a,
                     albedo=al,
                     h2o=h2o_scale,
                     o3=o3_scale,
                     co2=co2_scale)
    return file

def reduce_nc(infile,
              outfile="tmp/out.nc",
              opt_prop_file="/home/walther/Programms/ecrad-1.2.0/data/aerosol_cams_ifs_optics.nc"):
    with xr.open_dataset(infile) as ds:
        times = np.unique(ds.time.data)
        lats = np.unique(ds.latitude.data)
        lons = np.unique(ds.longitude.data)
        newshape=(len(times),len(lats),len(lons))
        zen = np.rad2deg(np.arccos(ds.cos_solar_zenith_angle.data.reshape(newshape)))

    with xr.open_dataset(opt_prop_file) as AERCFG:
        channels1_sw = 1./AERCFG.wavenumber1_sw[:]
        channels2_sw = 1./AERCFG.wavenumber2_sw[:]
        channels1_lw = 1./AERCFG.wavenumber1_lw[:]
        channels2_lw = 1./AERCFG.wavenumber2_lw[:]
        cwvl_sw=np.mean(np.vstack((channels1_sw,channels2_sw)),axis=0)[:-1]
        cwvl_sw*= 1e7 #[nm]
        cwvl_lw=np.mean(np.vstack((channels1_lw,channels2_lw)),axis=0)
        cwvl_lw*= 1e7 #[nm]
        
    newshape_wvl_sw = tuple(list(newshape)+[len(cwvl_sw)])
    newshape_wvl_lw = tuple(list(newshape)+[len(cwvl_lw)])

    with xr.open_dataset(outfile) as ds:
        dsn = xr.Dataset(dict(        
            flux_dn_sw_toa = (('time','lat','lon'),ds.flux_dn_sw.values[:,0].reshape(newshape)),
            flux_dn_sw_sfc = (('time','lat','lon'),ds.flux_dn_sw.values[:,-1].reshape(newshape)),
            flux_dn_direct_sw_sfc = (('time','lat','lon'),ds.flux_dn_direct_sw.values[:,-1].reshape(newshape)),
            flux_up_sw_toa = (('time','lat','lon'),ds.flux_up_sw.values[:,0].reshape(newshape)),
            flux_up_sw_sfc = (('time','lat','lon'),ds.flux_up_sw.values[:,-1].reshape(newshape)),
            flux_dn_lw_toa = (('time','lat','lon'),ds.flux_dn_lw.values[:,0].reshape(newshape)),
            flux_dn_lw_sfc = (('time','lat','lon'),ds.flux_dn_lw.values[:,-1].reshape(newshape)),
            flux_up_lw_toa = (('time','lat','lon'),ds.flux_up_lw.values[:,0].reshape(newshape)),
            flux_up_lw_sfc = (('time','lat','lon'),ds.flux_up_lw.values[:,-1].reshape(newshape)),
            spectral_flux_dn_sw_toa = (('time','lat','lon','wvl_sw'),
                                       ds.spectral_flux_dn_sw.values[:,0,:-1].reshape(newshape_wvl_sw)),
            spectral_flux_dn_sw_sfc = (('time','lat','lon','wvl_sw'),
                                       ds.spectral_flux_dn_sw.values[:,-1,:-1].reshape(newshape_wvl_sw)),
            spectral_flux_up_sw_toa = (('time','lat','lon','wvl_sw'),
                                       ds.spectral_flux_up_sw.values[:,0,:-1].reshape(newshape_wvl_sw)),
            spectral_flux_up_sw_sfc = (('time','lat','lon','wvl_sw'),
                                       ds.spectral_flux_up_sw.values[:,-1,:-1].reshape(newshape_wvl_sw)),
            spectral_flux_dn_lw_toa = (('time','lat','lon','wvl_lw'),
                                       ds.spectral_flux_dn_lw.values[:,0,:].reshape(newshape_wvl_lw)),
            spectral_flux_dn_lw_sfc = (('time','lat','lon','wvl_lw'),
                                       ds.spectral_flux_dn_lw.values[:,-1,:].reshape(newshape_wvl_lw)),
            spectral_flux_up_lw_toa = (('time','lat','lon','wvl_lw'),
                                       ds.spectral_flux_up_lw.values[:,0,:].reshape(newshape_wvl_lw)),
            spectral_flux_up_lw_sfc = (('time','lat','lon','wvl_lw'),
                                       ds.spectral_flux_up_lw.values[:,-1,:].reshape(newshape_wvl_lw)),
            spectral_flux_dn_direct_sw_sfc = (('time','lat','lon','wvl_sw'),
                                       ds.spectral_flux_dn_direct_sw.values[:,-1,:-1].reshape(newshape_wvl_sw)),
            spectral_flux_dn_direct_sw_toa = (('time','lat','lon','wvl_sw'),
                                       ds.spectral_flux_dn_direct_sw.values[:,0,:-1].reshape(newshape_wvl_sw)),
            sza = (('time','lat','lon'),zen)),
            coords= dict(time = times,
                         lat = lats,
                         lon = lons,
                         wvl_sw = cwvl_sw,
                         wvl_lw = cwvl_lw))
    return dsn

def run_ecrad(inputfile,namfile,outfile = False,reduce_out=True,
              ecrad_path="/home/walther/Programms/ecrad-1.2.0/bin/",
              opt_prop_file="/home/walther/Programms/ecrad-1.2.0/data/aerosol_cams_ifs_optics.nc"):
    ecrad_path = os.path.abspath(ecrad_path)
    # infile = os.path.relpath(inputfile,ecrad_path)
    # outfile = os.path.relpath(outfile,ecrad_path)
    nam = os.path.abspath("tmp/namfile.tmp")
    out = os.path.abspath("tmp/out.nc")
    infile = os.path.abspath("tmp/in.nc")
    
    # clear files
    if os.path.exists(nam): os.remove(nam)
    if os.path.exists(out): os.remove(out)
    if os.path.exists(infile): os.remove(infile)
    
    
    # store temp files
    # print("input:",inputfile.pressure_hl.shape)

    inputfile.to_netcdf(infile)
    with open(nam,'w') as txt:
        txt.write(namfile)


    # run ecrad
    command = "{ecrad} {namfile} {infile} {outfile}"
    
    cmd = command.format(ecrad=os.path.join(ecrad_path,'ecrad'),
                         namfile=nam,
                         infile=infile,
                         outfile=out)
    p = subprocess.Popen(cmd.split(),stdout=open('ecrad.log','w'))
    p.wait()
    
    
    # check succesful run
    if not os.path.exists(out):
        print(" ECRAD run fails! ")
        return None
    
    # store only selected output variables
    if reduce_out:
        dsn = reduce_nc(infile,out,opt_prop_file=os.path.join(os.path.abspath(ecrad_path),'../data/',opt_prop_file))
    else:
        dsn = xr.open_dataset(out).compute()
    # print("output:",dsn.flux_dn_sw.values[:,-1])
    # encode and write to desired location
    encoding={}
    for key in dsn.keys():
        encoding.update({key:{'zlib':True}})
    if outfile:
        outfile = os.path.abspath(outfile)
        dsn.to_netcdf(outfile,encoding=encoding)
        
    # return ecrad output
    return dsn

def get_ecrad_set(ifile,time=False,reduce_out=True,ecrad_path="/home/walther/Programms/ecrad-1.2.0/bin/",opt_prop_file="/home/walther/Programms/ecrad-1.2.0/data/aerosol_cams_ifs_optics.nc"):
    """
    Calculate aerosol radiative effect set of ecrad (with and without aerosol),
    at given date and station.

    Parameters
    ----------
    ifile : xr Dataset
        ecRad input file
    time : numpy array (float), optional
        If set, this variable will be used as main coordinate in output datasets.
        The default is False.
    reduce_out : bool
        if True ecRad output will be reduced to TOA and SFC fluxes only, default is True
    ecrad_path : string
        path to ecrad executable
    opt_prop_file: string
        path to aerosol opt. props file

    Returns
    -------
    ds1 : xarray dataset
        Ecrad output as xarray dataset (atmosphere + aerosol).
    ds2 : xarray dataset
        Ecrad output as xarray dataset (no aerosol)
    """
    if not type(time)==bool:
        ifile = interp_ds(ifile,time)
        
    nfile1 = make_namfile_str(ecrad_path=os.path.join(ecrad_path,'..'),
                              optpropfile=os.path.basename(opt_prop_file))
    nfile2 = make_namfile_str(aerosol=False,
                              ecrad_path=os.path.join(ecrad_path,'..'),
                              optpropfile=os.path.basename(opt_prop_file))
    ds1 = run_ecrad(inputfile=ifile,
                    namfile=nfile1,
                    ecrad_path=ecrad_path,
                    opt_prop_file=opt_prop_file,
                    reduce_out=reduce_out)
    ds2 = run_ecrad(inputfile=ifile, 
                    namfile=nfile2,
                    ecrad_path=ecrad_path,
                    opt_prop_file=opt_prop_file,
                    reduce_out=reduce_out)
    return ds1,ds2

def get_ecrad_ds(ifile,
                 outfile = False,
                 reduce_out = True,
                 perturbation=[],
                 aerosol=True,
                 h2o_scale=1,
                 o3_scale=1,
                 co2_scale=1,
                 time=False,
                 ecrad_path = "/home/walther/Programms/ecrad-1.2.0/bin/",
                 opfile = "aerosol_cams_ifs_optics.nc",
                 debug=False):
    """
    Calculate aerosol radiative effect set of ecrad (with and without aerosol),
    at given date and station.

    Parameters
    ----------
    ifile : xarray Dataset
        ecRad input file
    outfile : string
        path to store ecrad output, or False - no stored file, default is False
    reduce_out : bool
        if true, reduce ecrad output to toa and sfc fluxes only
    perturbation : list of tuples of shape (str key, (slice selection, float perturb[percent])), optional
        For each key in pertubation list, the input of key[selection] is pertubed with an relative increase 
        according to its pertub value. 'selection' has to be a tuple of The default is [] (no pertubation).
    aerosol : bool, optional
        If 'True' calculate ecrad with aerosols, elso not. The default is True.
    h2o_scale, o3_scale, co2_scale : float
        scale gases in ecrad
    time : numpy array (float), optional
        If set, this variable will be used as main coordinate in output datasets.
        The default is False.
    ecrad_path : string
        path to ecrad executable
    opt_prop_file: string
        path to aerosol opt. props file
    debug : bool
        if true, store perturbed aerosol file and input file in the working directory

    Returns
    -------
    ds : xarray dataset
        Ecrad output as xarray dataset.
    """
    
    ecrad_data_path=os.path.join(ecrad_path,'../data/')
    ## make a new (pertubed) optical properties file
    opfile = os.path.join(os.path.abspath(ecrad_data_path),opfile)
    tmp_opfile = opfile.replace('.nc',"_tmp.nc")
    # remove from previose run
    if os.path.exists(tmp_opfile): os.remove(tmp_opfile)
    # pertube optprop file
    with xr.open_dataset(opfile) as OP:
        varnames = [key for key in OP.keys()]
        for key,(select,pert) in perturbation:
            if not key in varnames:
                continue
            OP[key].values[select] = OP[key].values[select]*(1.+(pert/100.))
        OP.to_netcdf(tmp_opfile)
    
    # for investigation store optfiles in local dir
    if debug:
        if os.path.exists("debug_optprop_before_pertube.nc"): os.remove("debug_optprop_before_pertube.nc")
        if os.path.exists("debug_optprop_after_pertube.nc"): os.remove("debug_optprop_after_pertube.nc")
        shutil.copy(opfile,"debug_optprop_before_pertube.nc")
        shutil.copy(tmp_opfile,"debug_optprop_after_pertube.nc")

    ## input file
    # change main coordinate if nessesary
    if not type(time)==bool:
        ifile = interp_ds(ifile,time)
    # pertube infile
    varnames = [key for key in ifile.keys()]
    for key,(select,pert) in perturbation:
        if not key in varnames:
            continue
        if debug:
            if os.path.exists("debug_input_before_perturbe.nc"): os.remove("debug_input_before_perturbe.nc")
            ifile.to_netcdf("debug_input_before_perturbe.nc")
        ifile[key].values[select] = ifile[key].values[select]*(1.+(pert/100.))
        if debug:
            if os.path.exists("debug_input_after_perturbe.nc"): os.remove("debug_input_after_perturbe.nc")
            ifile.to_netcdf("debug_input_after_perturbe.nc")
    # make namfile
    nfile = make_namfile_str(aerosol=aerosol,
                             ecrad_path=os.path.join(ecrad_path,'..'),
                             optpropfile=os.path.basename(tmp_opfile),
                             h2o_scale = h2o_scale,
                             co2_scale = co2_scale,
                             o3_scale = o3_scale)
    # calculate with ecrad
    ds = run_ecrad(inputfile=ifile,
                   outfile = outfile,
                   reduce_out=reduce_out,
                   namfile=nfile,
                   ecrad_path=ecrad_path,
                   opt_prop_file=tmp_opfile)
    return ds


def interp_ds(ds,newtimes,newlats=False,newlons=False):
    oldtimes = np.unique(ds.time.values)
    if type(newtimes) == bool:
        newtimes = oldtimes
    oldlats = np.unique(ds.latitude.values)
    if type(newlats) == bool:
        newlats = oldlats
    oldlons = np.unique(ds.longitude.values)
    if type(newlons) == bool:
        newlons = oldlons
    
    dims = ('time','lat','lon')
    shape = (len(oldtimes),len(oldlats),len(oldlons))
    
    # if only for one stations, take the short route
    if shape[1]==1 and shape[2]==1: 
        # make time the main coordinate 
        wds = ds.assign_coords({'time':ds.time}).swap_dims({'col':'time'})
        wds = wds.drop(['col'])
        # drop mu0 since we dont want to interpolate it
        wds = wds.drop(['cos_solar_zenith_angle'])
        # interpolate on the time axis with new time index
        wds = wds.interp(dict(time=newtimes),assume_sorted=True)
        # reassign col as main coordinate
        wds = wds.assign_coords({'col':('time',np.arange(len(wds.time)))}).swap_dims({'time':'col'}).dropna('col')
        
    else:
        # drop time, lat, lon variables which have 'col' dimension
        wds = ds.drop_vars(['time','latitude','longitude'])
        # drop also mu0 since we dont want to interpolate it
        wds = wds.drop(['cos_solar_zenith_angle'])
        
        # assigne the unique values of time, lat, lon as coordinates
        wds = wds.assign_coords({'time':('time',oldtimes),
                                 'lat':('lat',oldlats),
                                 'lon':('lon',oldlons),
                                 })
        # overwrite all variables with 'col' dimesion to (time,lat,lon) dims
        for key in wds.keys():
            olddims = wds[key].dims
            oldshape = wds[key].shape
            if len(olddims)==0:
                continue
            if olddims[0] == 'col':
                newdims = tuple(list(dims) + list(olddims[1:]))
                newshape = tuple(list(shape) + list(oldshape[1:]))
            else:
                continue
            wds = wds.assign({f"{key}": (newdims,wds[key].values.reshape(newshape))})

        # col dim can now be dropped
        wds = wds.drop_dims('col')

        # finally interpolate to new values
        wds = wds.interp({'time':newtimes,
                          'lat':newlats,
                          'lon':newlons})

        # reindex to 'col' struckture
        newcol = np.arange(len(newtimes)*len(newlats)*len(newlons))

        # assign new col coordinates
        wds = wds.assign_coords({'col':newcol})

        # reshape to col dims
        for key in wds.keys():
            olddims = wds[key].dims
            oldshape = wds[key].shape
            if len(olddims)==0:
                continue
            if olddims[0] == 'time':
                newdims = tuple(['col'] + list(olddims[3:]))
                newshape = tuple([len(newcol)] + list(oldshape[3:]))
            else:
                continue
            wds = wds.assign({f"{key}": (newdims,wds[key].values.reshape(newshape))})
        # drop time,lat,lon dims
        wds = wds.drop_dims(['time','lat','lon'])

        # assign flattened time, lat ,lon variables
        times,lats,lons = np.meshgrid(newtimes,newlats,newlons,indexing='ij')
        times = times.flatten()
        lats = lats.flatten()
        lons = lons.flatten()
        wds = wds.assign({'time': ('col',times),
                          'latitude': ('col',lats),
                          'longitude': ('col',lons)})
        wds = wds.dropna('col')
    
    # calculate cosine of zenith angle for new times
    # jd = sp.datetime2julday(wds.time.values)
    # sza, azi = sp.zenith_azimuth(jd, np.nanmean(wds.latitude.values), np.nanmean(wds.longitude.values))
    sza,azi = sp.sun_angles(wds.time.values,
                            np.nanmean(wds.latitude.values),
                            np.nanmean(wds.longitude.values))
    mu0 = np.cos(np.deg2rad(sza))
    # reassign mu0 to dataset
    wds = wds.assign({'cos_solar_zenith_angle':('col',mu0)})
    return wds
