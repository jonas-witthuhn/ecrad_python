#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:26:01 2020

@author: walther
"""
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d

# Hartwigs sunpos routine
from trosat import sunpos as sp
# import sunpos as sp

def load_tsi(pf="../data/sorce_tsi_24hr_l3_full.csv"):
    """ Load the SORCE TSI dataset and read it to an xarray dataset.
    E0 - solar irradiance at earth orbit is calculated additionally.
    """
    df=pd.read_csv(pf,parse_dates=[0],date_parser=lambda x: x.astype("datetime64[D]"))
    ds=xr.Dataset.from_dataframe(df)
    for key in ds.keys():
        newkey=key.split()[0]
        ds=ds.rename_vars({key:newkey})
    ds=ds.swap_dims({'index':'time'})
    #calculate earth sun distance
    # ESD=sp.earth_sun_distance(sp.datetime2julday(ds.time))
    ESD = sp.earth_sun_distance(ds.time.values)
    ds = ds.assign(E0=ds['tsi_1au']/(ESD**2))
    # fill missing days
    ds = ds.reindex({'time':pd.date_range(ds.time.values[0],ds.time.values[-1],freq='1D')},method='nearest')
    return ds

def calc_lvl_pressure(A, B, p0=1013.25):
    """calculate model level pressure (interface or half level) from defined 
    constants A and B from grib metadata. Pk = Ak [Pa] + p0 [hPa]*Bk"""
    if np.isscalar(p0):
        P = (A/100.) + p0*B
    else:
        P = (A/100.) + p0[:,np.newaxis]*B
    return P

def flatten_coords(A, idx_keep=1):
    """ flattening the data of an array but keeping one axis.
    Example:
        >>> a = np.array([[1,2,3],
                          [4,5,6]])
        >>> a.flatten()
            array([1, 2, 3, 4, 5, 6])
        >>> A = np.zeros((a.shape[0],2,a.shape[1]))
        >>> A[:,0,:] = a.copy()
        >>> A[:,1,:] = a*10
        >>> flatten_coords(A,1)
            array([[ 1., 10.],
                   [ 2., 20.],
                   [ 3., 30.],
                   [ 4., 40.],
                   [ 5., 50.],
                   [ 6., 60.]])
    """
    N = A.shape[idx_keep]
    B = np.zeros(A.reshape(-1,N).shape)
    for i in range(N):
        B[:,i] = A.take(i,axis=idx_keep).flatten()
    return B

def prepare_ecrad_input(date,
                        ncpath = 'data/nc/',
                        outpath = 'data/ecrad/',
                        sorce_tsi_fname = "../data/sorce_tsi_24hr_l3_full.csv",
                        outfile_fname_temp = "input_{date:%Y%m%d}.nc",
                        camsra_fname_temp = "cams-ra_{date:%Y-%m-%d}_{levtype}.nc",
                        cams73_fname_temp = "cams73_latest_{gas}_col_surface_inst_{date:%Y%m}.nc"):
    """ prepare input netCDF file for ECRAD
    required CAMS data:
        [
            {
                "levtype"  : "ml",
                "levelist" : "1/to/60",
                "param"    : [
                    "129",        // Geopotential
                    "130",        // Temperature
                    "131",        // U component of wind
                    "132",        // V component of wind
                    "135",        // Vertical velocity
                    // atmospheric gases
                    "133",        // Specific humidity
                    "210121",    // Nitrogen dioxide
                    "210203",    // GEMS Ozone
                    // Aerosol mixing ratios
                    "210048",    // Aerosol large mode mixing ratio
                    "210001",    // Sea salt aerosol mr (0.03 - 0.5um)
                    "210002",    // Sea salt aerosol mr (0.55 - 0.9um)
                    "210003",    // Sea salt aerosol mr (0.9 - 20um)
                    "210004",    // Dust Aerosol mr (0.03-0.55um)
                    "210005",    // Dust Aerosol mr (0.55 - 0.9um)
                    "210006",    // Dust Aerosol mr (0.9 - 20um)
                    "210007",    // hydrophilic organic matter aerosol mr 
                    "210008",    // hydrophobic organic matter aerosol mr
                    "210009",    // hydrophilic black carbon aerosol mr
                    "210010",    // hydrophobic black carbon aerosol mr
                    "210011"    // Sulphate aerosol mr 
                ]
            },
            {
                "levtype"  : "sfc",
                "param"    : [
                    "167.128", // 2 metre temperature
                    "168.128", // 2m dewpoint temperature
                    "129.128", // geopotential/elevation
                    "134.128", // surface pressure
                    "235.128", // skin temperature
                    // AOD for method comparison
                    "213.210", // Total Aerosol Optical Depth at 469nm
                    "207.210", // Total Aerosol Optical Depth at 550nm
                    "214.210", // Total Aerosol Optical Depth at 670nm
                    "215.210", // Total Aerosol Optical Depth at 865nm
                    "216.210", // Total Aerosol Optical Depth at 1240nm
                    // Albedo
                    "15.128", // UV visible albedo for direct radiation
                    "16.128", // UV visible albedo for diffuse radiation
                    "17.128", // Near-ir albedo for direct radiation
                    "18.128" // Near-ir albedo for diffuse radiation
                ]
            }
        ]
        ## Additional CO2 and CH4 total columne volumne mixing ratio is taken
        ## from CAMS73 (https://atmosphere.copernicus.eu/index.php/cams73-greenhouse-gases-fluxes)
    """
    date=pd.to_datetime(date)
    # assign file names
    fname_ml = camsra_fname_temp.format(date=date, levtype='ml')
    fname_sfc = camsra_fname_temp.format(date=date, levtype='sfc')
    fname_co2 = cams73_fname_temp.format(date=date, gas='co2')
    fname_ch4 = cams73_fname_temp.format(date=date, gas='ch4')
    # open datasets
    with xr.open_dataset(os.path.join(ncpath,fname_ml)) as ds_ml, \
         xr.open_dataset(os.path.join(ncpath,fname_sfc)) as ds_sfc:
        
        ## remove nan from psfc 
        ds_sfc = ds_sfc.dropna('lat',how='all',subset=['psfc'])
        ds_sfc = ds_sfc.dropna('lon',how='all',subset=['psfc'])
        ds_ml = ds_ml.reindex_like(ds_sfc)
        
        
        ## Prepare ECRAD cols coordinate
        # flatten time, lat, lon coordinates, so that each cols[i] correspond
        # to a uniqe set of them.
        times, lats, lons = np.meshgrid(ds_ml.time,ds_ml.lat, ds_ml.lon,indexing = 'ij')
        times = times.flatten()
        lats = lats.flatten()
        lons = lons.flatten()
        cols = np.arange(len(lons))

        ## Prepare ECRAD level and half-level coordinates
        # level - mid point of each layer -> CAMS-RA [nhym] coordinate
        # half-level - layer interfaces + surface and TOA -> CAMSRA [nhyi] coordinate
        level = ds_ml.nhym.data
        half_level = ds_ml.nhyi.data

        ## calculate cosine of zenith angle
        # jd = sp.datetime2julday(times)
        # sza, azi = sp.zenith_azimuth(jd, lats, lons)
        sza, azi = sp.sun_angles(times,lats,lons)
        mu0 = np.cos(np.deg2rad(sza))

        ## get F0 - solar irradiance at earth orbit
        ds_tsi = load_tsi(sorce_tsi_fname)
        F0 = float(ds_tsi.E0.sel({'time':date.strftime("%Y-%m-%d")}).data)

        ## sw_albedo
        # CAMS-RA shortwave albedo at two spectral bands:
        # aluvp and aluvd 0.3 - 0.7um
        # alnip and alnid 0.7 - 5um
        try:
            aluvp = ds_sfc.aluvp.data.flatten()
            alnip = ds_sfc.alnip.data.flatten()
            sw_albedo_direct = np.vstack((aluvp,alnip)).T
            aluvd = ds_sfc.aluvd.data.flatten()
            alnid = ds_sfc.alnid.data.flatten()
            sw_albedo = np.vstack((aluvd,alnid)).T
        except:
            sw_albedo = np.ones((len(cols),2))*0.1
            sw_albedo_direct = np.ones((len(cols),2))*0.1
        ## lw_emissivity
        # choose a fixed value according to Bellouin et al 2020.
        # (https://doi.org/10.5194/essd-2019-251)
        Eland = 0.96
        Esea = 0.99
        try:
            lsm = ds_sfc.lsm.data.flatten()
            lw_emissivity =Esea*(1-lsm) + Eland*lsm # weight values with land-sea-mask
            lw_emissivity = lw_emissivity[:,np.newaxis]
        except:
            lw_emissivity = np.ones(len(cols))*Eland
            lw_emissivity = lw_emissivity[:,np.newaxis]

        ## calculate pressure between levels including sfc and TOA
        p0 = ds_sfc.psfc.data.flatten() # surface pressure
        # pressure at half-level -> shape(col, half_level)
        pressure_ilvl = calc_lvl_pressure(ds_ml.hyai.data,ds_ml.hybi.data,p0)
        # pressure at level -> shape(col, level)
        pressure_mlvl = calc_lvl_pressure(ds_ml.hyam.data,ds_ml.hybm.data,p0)
        
        ## calculate temperature at at level interfaces
        # flatten all coordinates as before, but saving the level dimension
        T_mlvl = flatten_coords(ds_ml.t.data,1) # shape(cols,level)
        try:
            T_0 = ds_sfc.tsfc.data.flatten()
        except:
            T_0 = ds_sfc.t2m.data.flatten()
        T_ilvl = np.zeros((len(cols),len(half_level)))
        for i in range(len(cols)):
            X = pressure_mlvl[i,:]
            Y = T_mlvl[i,:]
            # linear interpolate between layer mids (pressure)
            f_interp = interp1d(X,Y)
            T_ilvl[i,1:-1] = f_interp(pressure_ilvl[i,1:-1])
            # add values for TOA and sfc
            T_ilvl[i,0] = T_mlvl[i,0] # T at TOA, use closest level value
            T_ilvl[i,-1] = T_0[i] # T at sfc

        ## Aerosol mixing ratios
        # CAMSRA Aerosols:
        # Sea salt 0.03-0.5             OPAC
        # Sea salt 0.5-5.               OPAC
        # Sea salt 5-20                 OPAC
        # Dust 0.03-0.55                Dubovic et al 2002
        # Dust 0.55-0.9                 Woodward et al 2001
        # Dust 0.9-20                   Fouquart et al 1987
        # Organic Matter hydrophilic    OPAC-Mixture
        # Organic Matter hydrophobic    OPAC Mixture at 20% humidity
        # Black Carbon hydrophilic      OPAC (SOOT) (not implemented and
        #                            treatet independed from relative humidity)
        # Black Carbon hydrophobic      OPAC (SOOT)
        # Sulfates (hydrophilic)        Lacis et al (GACP)
        ## flattening again all coordinates but level
        MMR_aer = np.zeros((len(cols),11,len(level)))
        MMR_aer[:,0,:] = flatten_coords(ds_ml.aermr01.data,1)
        MMR_aer[:,1,:] = flatten_coords(ds_ml.aermr02.data,1)
        MMR_aer[:,2,:] = flatten_coords(ds_ml.aermr03.data,1)
        MMR_aer[:,3,:] = flatten_coords(ds_ml.aermr04.data,1)
        MMR_aer[:,4,:] = flatten_coords(ds_ml.aermr05.data,1)
        MMR_aer[:,5,:] = flatten_coords(ds_ml.aermr06.data,1)
        MMR_aer[:,6,:] = flatten_coords(ds_ml.aermr07.data,1)
        MMR_aer[:,7,:] = flatten_coords(ds_ml.aermr08.data,1)
        MMR_aer[:,8,:] = flatten_coords(ds_ml.aermr09.data,1)
        MMR_aer[:,9,:] = flatten_coords(ds_ml.aermr10.data,1)
        MMR_aer[:,10,:] = flatten_coords(ds_ml.aermr11.data,1)
        MMR_aer[MMR_aer<0]=0

        ## gas mixing ratios at full level
        MMR_h2o = flatten_coords(ds_ml.q.data,1)
        MMR_h2o[MMR_h2o<0] = 0
        MMR_o3 = flatten_coords(ds_ml.go3.data,1)
        MMR_o3[MMR_o3<0] = 0
        ## volume mixing ratios of more trace gases
        # have to be converted to volumne mixing ratio
        Mm_dry = 28.9645 # molar mass dry air
        Mm_no2 = 46.0055
        # Mm_co = 28.0101
        # Mm_ch4 = 16.0425
        # Mm_co2 = 44.0095
        # Mm_so2 = 64.0638

        # --- could be extended with additionaltrace gas data
        variables = {'solar_irradiance':F0,
                        'skin_temperature':('col',T_0),
                        'cos_solar_zenith_angle': ('col',mu0),
                        'sw_albedo': (('col','sw_albedo_band'),sw_albedo),
                        'sw_albedo_direct': (('col','sw_albedo_band'),sw_albedo_direct),
                        'lw_emissivity': (('col','lw_emiss_band'),lw_emissivity),
                        'pressure_hl': (('col','half_level'),pressure_ilvl),
                        'temperature_hl': (('col','half_level'),T_ilvl),
                        'h2o_mmr': (('col','level'),MMR_h2o),
                        'o3_mmr': (('col','level'),MMR_o3),
                        'aerosol_mmr': (('col','aerosol_type','level'),MMR_aer),
                        'time':('col',times),
                        'latitude': ('col',lats),
                        'longitude':('col',lons)}
        try:
            MMR_no2 = flatten_coords(ds_ml.no2.data,1)
            VMR_no2 = (Mm_dry/Mm_no2) * MMR_no2
            VMR_no2[VMR_no2<0] = 0
            variables.update({"no2_vmr":VMR_ch4})
        except:
            pass
            
        
        
        try:
            ds_co2 = xr.open_dataset(os.path.join(ncpath,fname_co2))
        except:
            ds_co2 = False
        if not type(ds_co2)==bool:
            ## Co2 -- only available from cams73
            # (https://atmosphere.copernicus.eu/index.php/cams73-greenhouse-gases-fluxes)
            idxlat = (ds_co2.latitude>=np.min(lats))*(ds_co2.latitude<=np.max(lats))
            idxlon = (ds_co2.longitude>=np.min(lons))*(ds_co2.longitude<=np.max(lons))
            VMR_co2= ds_co2.XCO2.sel({'time':date.strftime('%Y-%m-%d'),
                                      'latitude':ds_co2.latitude.data[idxlat],
                                      'longitude':ds_co2.longitude.data[idxlon]})
            # calculate mean over domain
            VMR_co2 = float(np.mean(VMR_co2)*1e-6)
            if VMR_co2<0:
                VMR_co2=0
            variables.update({"co2_vmr":VMR_co2})
        
        try:
            ds_ch4 = xr.open_dataset(os.path.join(ncpath,fname_ch4))
        except:
            ds_ch4=False

        if not type(ds_ch4) == bool:
            ## CH4 --would also be available from CAMS-RA but was included in the 
            # CAMS73 data order
            idxlat = (ds_ch4.latitude>=np.min(lats))*(ds_ch4.latitude<=np.max(lats))
            idxlon = (ds_ch4.longitude>=np.min(lons))*(ds_ch4.longitude<=np.max(lons))
            VMR_ch4= ds_ch4.CH4.sel({'time':date.strftime('%Y-%m-%d'),
                                     'latitude':ds_ch4.latitude.data[idxlat],
                                     'longitude':ds_ch4.longitude.data[idxlon]})
            # calculate mean over domain
            VMR_ch4 = float(np.mean(VMR_ch4)*1e-9)
            if VMR_ch4<0:
                VMR_ch4=0
            variables.update({"ch4_vmr":VMR_ch4})
    ## Fill ECRAD dataset
    ecrad = xr.Dataset(variables,
                        coords={'col':('col',cols),
                                'half_level':('half_level',half_level),
                                'level':('level',level),
                                'aerosol_type':('aerosol_type',np.arange(11)),
                                'sw_albedo_band': ('sw_albedo_band',np.arange(2)),
                                'lw_emiss_band': ('lw_emiss_band',np.arange(1))})
    ## enable zlib for all variables to compress netcdf file
    encoding={}
    for key in ecrad.keys():
        encoding.update({key:{'zlib':True}})
    ## store dataset as netCDF
    ecrad.to_netcdf(path = os.path.join(outpath,outfile_fname_temp.format(date=date)),
                    encoding = encoding)
    return 0

if __name__=='__main__':
    dates = np.arange("2015-01-01","2016-01-01",dtype="datetime64[D]")
    for date in dates:
        prepare_ecrad_input(date = date,
                        ncpath = 'data/nc/',
                        outpath = 'data/ecrad/',
                        outfile_fname_temp = "input_{date:%Y%m%d}.nc",
                        camsra_fname_temp = "cams-ra_{date:%Y-%m-%d}_{levtype}.nc",
                        cams73_fname_temp = "cams73_latest_{gas}_col_surface_inst_{date:%Y%m}.nc")
