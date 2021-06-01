import numpy as np
import xarray as xr

def calc_AE(t1,t2,l1,l2):
    return -np.log(t1/t2)/np.log(l1/l2)

def calc_OD(l,ae,tref,lref):
    return tref*((l/lref)**(-ae))

def perturbe_AE(OPfile,pert):
    """
    to variate the AE in the aerosol optical props file for the ECRAD input.
    The mass extinction coefficients (mext) has to be changed accordingly.
    mext is a function of wavelength (wvl) and relative_humidity (rh) for
    hydrophilic aerosol types. In this function, we loop over all mext
    calculating the new mext as it would be with a different AE (440/870nm).
    
    This is done for the "mono" variables and the band channel variables.
    
    Parameters
    ----------
    OPfile : string
	path to the netcdf aerosol optical propeties file
    pert : float
	amount of pertubation in percent. for example pert=1 -> the newAE  = oldAE + oldAE*pert/100
	
    Returns
    -------
    pertubations : list
	list of perturbation input for the run_ecrad.get_ecrad_ds module:
	    list of tuples of shape (str key, (slice selection, float pertub[percent]))
	    For each key in pertubation list, the input of key[selection] is pertubed with an relative increase 
	    according to its pertub value. 'selection' has to be a slice for the selection to be perturbet.
    """
  
  
    pertubations=[]
    with xr.open_dataset(OPfile) as OP:
        # mono
        wvls = np.round(OP.wavelength_mono.values * 10**9,0) #[nm]
        lref = 550. # reference wavelength [nm]
        iref = np.argwhere(wvls==lref)[0][0]
        l1 = 440.
        l2 = 865.
        i1 = np.argwhere(wvls==l1)[0][0]
        i2 = np.argwhere(wvls==l2)[0][0]
        # for hydrophilic
        key = 'mass_ext_mono_hydrophilic'
        AE = calc_AE(OP[key][:,:,i1],OP[key][:,:,i2],l1,l2)
        AEp = AE*(1+pert/100.)

        AOD = calc_OD(wvls[np.newaxis,np.newaxis,:],AE.values[:,:,np.newaxis],OP[key].values[:,:,iref][:,:,np.newaxis],lref)
        AODp = calc_OD(wvls[np.newaxis,np.newaxis,:],AEp.values[:,:,np.newaxis],OP[key].values[:,:,iref][:,:,np.newaxis],lref)
        for i in range(len(OP.hydrophilic)):
            for j in range(len(OP.relative_humidity)):
                for k in range(len(wvls)):
                    pertubations.append((key,((i,j,k),100.*((AODp[i,j,k]/AOD[i,j,k])-1))))

        # for hydrophobic
        key = 'mass_ext_mono_hydrophobic'
        AE = calc_AE(OP[key][:,i1],OP[key][:,i2],l1,l2)
        AEp = AE*(1+pert/100.)

        AOD = calc_OD(wvls[np.newaxis,:],AE.values[:,np.newaxis],OP[key].values[:,iref][:,np.newaxis],lref)
        AODp = calc_OD(wvls[np.newaxis,:],AEp.values[:,np.newaxis],OP[key].values[:,iref][:,np.newaxis],lref)
        for i in range(len(OP.hydrophobic)):
            for k in range(len(wvls)):
                pertubations.append((key,((i,k),100.*((AODp[i,k]/AOD[i,k])-1))))


        # bands sw
        channels1 = 1./OP.wavenumber1_sw[:]
        channels2 = 1./OP.wavenumber2_sw[:]
        wvls=np.mean(np.vstack((channels1,channels2)),axis=0)
        wvls*= 1e7 #[nm]

        iref = 9
        i1 = 10
        i2 = 8

        lref = wvls[iref]
        l1 = wvls[i1]
        l2 = wvls[i2]

        # for hydrophilic
        key = 'mass_ext_sw_hydrophilic'
        AE = calc_AE(OP[key][:,:,i1],OP[key][:,:,i2],l1,l2)
        AEp = AE*(1+pert/100.)

        # sw
        AOD = calc_OD(wvls[np.newaxis,np.newaxis,:],AE.values[:,:,np.newaxis],OP[key].values[:,:,iref][:,:,np.newaxis],lref)
        AODp = calc_OD(wvls[np.newaxis,np.newaxis,:],AEp.values[:,:,np.newaxis],OP[key].values[:,:,iref][:,:,np.newaxis],lref)
        for i in range(len(OP.hydrophilic)):
            for j in range(len(OP.relative_humidity)):
                for k in range(len(wvls)):
                    pertubations.append((key,((i,j,k),100.*((AODp[i,j,k]/AOD[i,j,k])-1))))

        # lw
        channels1 = 1./OP.wavenumber1_lw[:]
        channels2 = 1./OP.wavenumber2_lw[:]
        wvls=np.mean(np.vstack((channels1,channels2)),axis=0)
        wvls*= 1e7 #[nm]
        AOD = calc_OD(wvls[np.newaxis,np.newaxis,:],AE.values[:,:,np.newaxis],OP[key].values[:,:,iref][:,:,np.newaxis],lref)
        AODp = calc_OD(wvls[np.newaxis,np.newaxis,:],AEp.values[:,:,np.newaxis],OP[key].values[:,:,iref][:,:,np.newaxis],lref)
        for i in range(len(OP.hydrophilic)):
            for j in range(len(OP.relative_humidity)):
                for k in range(len(wvls)):
                    pertubations.append(('mass_ext_lw_hydrophilic',((i,j,k),100.*((AODp[i,j,k]/AOD[i,j,k])-1))))


        # sw
        channels1 = 1./OP.wavenumber1_sw[:]
        channels2 = 1./OP.wavenumber2_sw[:]
        wvls=np.mean(np.vstack((channels1,channels2)),axis=0)
        wvls*= 1e7 #[nm]

        iref = 9
        i1 = 10
        i2 = 8

        lref = wvls[iref]
        l1 = wvls[i1]
        l2 = wvls[i2]


        # for hydrophobic
        key = 'mass_ext_sw_hydrophobic'
        AE = calc_AE(OP[key][:,i1],OP[key][:,i2],l1,l2)
        AEp = AE*(1+pert/100.)

        AOD = calc_OD(wvls[np.newaxis,:],AE.values[:,np.newaxis],OP[key].values[:,iref][:,np.newaxis],lref)
        AODp = calc_OD(wvls[np.newaxis,:],AEp.values[:,np.newaxis],OP[key].values[:,iref][:,np.newaxis],lref)
        for i in range(len(OP.hydrophobic)):
            for k in range(len(wvls)):
                pertubations.append((key,((i,k),100.*((AODp[i,k]/AOD[i,k])-1))))

        # lw
        channels1 = 1./OP.wavenumber1_lw[:]
        channels2 = 1./OP.wavenumber2_lw[:]
        wvls=np.mean(np.vstack((channels1,channels2)),axis=0)
        wvls*= 1e7 #[nm]
        AOD = calc_OD(wvls[np.newaxis,:],AE.values[:,np.newaxis],OP[key].values[:,iref][:,np.newaxis],lref)
        AODp = calc_OD(wvls[np.newaxis,:],AEp.values[:,np.newaxis],OP[key].values[:,iref][:,np.newaxis],lref)
        for i in range(len(OP.hydrophilic)):
            for k in range(len(wvls)):
                pertubations.append(('mass_ext_lw_hydrophopic',((i,k),100.*((AODp[i,k]/AOD[i,k])-1))))

    return pertubations

# def calc_kernel()

# def unpertubed_run()
