import numpy as np
from scipy.interpolate import UnivariateSpline as USpline
from astropy.io import fits
import glob

data_paths = '../data/*.fits'

def get_name(name):
    if name is None:
        return ''
    else:
        return ' - ' + name

# Cropping functions
def get_balmer(max_line):
	'''
	Generates the first `max_line` Balmer lines in units of Angstroms.
	'''
    balmer = (np.arange(3, max_line+1))**2
    balmer = 3645.07 * (balmer/(balmer-4))
    return balmer

def crop_lines(lmbd, flux, ivar, num_lines=9, crop_wdth=100, mean_wdth=0, start=8, step_size=25):
	'''
	Given a wavelength, flux, and inverse variance array, crops out the Balmer lines and samples
	a certain number of points to prepare the arrays for spline fitting.

	num_lines - the last Balmer line to directly crop, all data points bluewards of this line
				are ignored. num_lines must be an int >= 3.
	crop_wdth - the number of indices to crop out on both sides of each Balmer line (note that
				2*crop_wdth+1 indices will be ignored). If `crop_wdth` is an array, the crop
				width for each line is set individually.
	mean_wdth - the number of indices to collect on both sides of each selected index to take
				the mean of (note that 2*mean_wdth+1 indices will be used). mean_wdth is an int.
	start	  - the index in the wavelength array to start sampling at. All indices less than
				start are ignored.
	step_size - the step size for the sampling of indices. only one index every step_size indies
				will be included in the output arrays.
	'''
    line_idxs = np.searchsorted(lmbd, get_balmer(num_lines))
    if type(crop_wdth) is int:
        crop_wdth = [crop_wdth] * len(line_idxs)
    elif len(crop_wdth) == 1:
        crop_wdth = [crop_wdth[0]] * len(line_idxs)

    mask = np.zeros(len(lmbd), dtype=bool)
    mask[start::step_size] = True
    mask[np.argwhere(ivar<0.1)] = False
    mask[:line_idxs[-1]] = False
    for cdx, wdth in zip(line_idxs, crop_wdth):
        lo = cdx-wdth
        hi = cdx+wdth
        mask[lo:hi] = False
    
    keep_idxs = np.concatenate(np.argwhere(mask))
    lmbd_crop = np.zeros(keep_idxs.shape)
    flux_crop = np.zeros(keep_idxs.shape)
    ivar_crop = np.zeros(keep_idxs.shape)
    for i in range(len(keep_idxs)):
        keep = keep_idxs[i]
        lmbd_crop[i] = np.mean(lmbd[keep-mean_wdth:keep+mean_wdth+1])
        flux_crop[i] = np.mean(flux[keep-mean_wdth:keep+mean_wdth+1])
        ivar_crop[i] = np.sqrt(np.sum(ivar[keep-mean_wdth:keep+mean_wdth+1]**2))
		

    return lmbd_crop, flux_crop, ivar_crop


# Continuum normalization functions
def get_spline(lmbd, flux, ivar=None, degree=1, smooth=100):
	'''
	Returns an interpolating spline on the input wavelength and flux arrays.

	ivar   - the inverse variance for each flux measurement. Used as spline knot weights.
	degree - the maximum degree of the interpolating spline polynomials.
	smooth - the smoothing factor to be used in the interpolating spline.
	'''
    return USpline(lmbd, flux, w=ivar, k=degree, s=smooth, ext=0)

def get_curves(spline, lmbd, flux):
	'''
	Returns the continuum and normalized curves for the given wavelength and flux values under 
	the given spline.
	'''
    cont = spline(lmbd)
    norm = flux/cont
    return cont, norm

def clip_norm(norm, b_lim=0, u_lim=2):
	'''
	Clips values in the normalized array that extend beyond the top or bottom limit to the
	central value of 1.

	Warning: modifies the input array in place!
	'''
    norm[np.argwhere(norm>u_lim)] = 1
    norm[np.argwhere(norm<b_lim)] = 1
    return norm

def cont_norm(lmbd, flux, ivar, num_lines=9, wndw_init=100, wndw_step=0, mean_wdth=0, spl_2=False):
	'''
	The primary function used to interpolate a continuum and normalize the flux. Returns the computed
	spline for further interpolation, and the continuum and normalized fluxes at the same wavelengths
	as the input wavelength array.

	num_lines - the total number of lines to crop out of the flux. See get_balmer() for more details.
	wndw_init - the first window size for the cropping. If wndw_step is equal to zero, wndw_init is
				used as the window size for all Balmer lines.
	wndw_step - the iterative change in size for the cropping windows around the Balmer lines.
	mean_wdth - the width around each selected index for averaging. See crop_lines() for more details.
	spl_2	  - a boolean to dictate whether a second spline should be applied on the spline directly
				computed from the input arrays.
	'''
    if wndw_step != 0:
        decr_wndw = np.arange(wndw_init, wndw_init + num_lines*wndw_step, wndw_step)
    else:
        decr_wndw = wndw_init
    lmbd_crop, flux_crop, ivar_crop = crop_lines(lmbd, flux, ivar, num_lines=num_lines, crop_wdth=decr_wndw, mean_wdth=mean_wdth)
    spl = get_spline(lmbd_crop, flux_crop, ivar_crop)
    cont, norm = get_curves(spl, lmbd, flux)
    
    if spl_2:
        spl = get_spline(lmbd, cont, None)
        cont, norm = get_curves(spl, lmbd, flux)
    
    norm = clip_norm(norm)

	return spl, cont, norm


if __name__ == '__main__':
	files = glob.glob(data_paths)
	fname = files[0]
	with fits.open(fname) as f:
		flux = f[1].data['flux']
		lmbd = 10**f[1].data['loglam']
		ivar = f[1].data['ivar']
	
	spl, cont, norm = cont_norm(lmbd, flux, ivar)
	print(spl)