'''
Pipeline for spectrum preprocessing.
'''
import argparse
import numpy as np
import lmfit as lm
import lmfit.models as mods
from scipy.interpolate import UnivariateSpline as USpline
from scipy.interpolate import interp1d as interp
from collections import namedtuple
from astropy.io import fits
import glob
import matplotlib.pyplot as plt

class Spectrum:
	def __init__(self, id, lmbd, flux, ivar):#, snr):
		self.id = id
		self.lmbd = lmbd # linear support only, for now
		self.flux = flux
		self.ivar = ivar # output ivars not supported
		# self.snr = snr # signal-to-noise ratio not supported yet

def preprocess(spectra, pixel_frac_thresh=0.25, pixel_qual_thresh=0.5):#, snr_thresh=10):
	'''
	Pre-processes spectra by filtering by defined properties.
	'''
	filtered = []

	for s in spectra:
		# if s.snr >= snr_thresh:
		num_bad = len(np.where(s.ivar <= pixel_qual_thresh, s.ivar, 0).nonzero())
		if num_bad / len(s.lmbd) < pixel_frac_thresh:
			filtered.append(s)

	return filtered

def redshift(spectra):
	'''
	Applies and corrects for radial-velocity redshift in the spectra.
	Modifies `spectrum.lmbd`.
	'''
	def lin_res(params, x, data, ivar):
		'''
		Returns the weighted residuals of a given linear model and the real data.
		'''
		return (data-lin_model(x, params['m'], params['b']))*np.sqrt(ivar)

	def lin_model(x, m, b):
		return m*x + b

	def fit_cont(lmbd, flux, ivar):
		'''
		Given a small section of spectrum around an absorption line, fits the local linear continuum and normalization.
		'''
		# idxs = np.arange(len(lmbd))
		# idxs = np.asarray(idxs < len(lmbd)/4 or idxs > 3*len(lmbd)/4).nonzero()
		idxs = np.concatenate((np.arange(0,len(lmbd)//4), np.arange(3*len(lmbd)//4,len(lmbd))))
		
		lpar = lm.Parameters()
		lpar.add('b', value=flux[0], vary=True)
		lpar.add('m', value=0, vary=True)
		lfit = lm.minimize(lin_res, lpar, args=(lmbd[idxs], flux[idxs], ivar[idxs]))
		m = lfit.params['m'].value
		b = lfit.params['b'].value
		
		flux_cont = lin_model(lmbd, m, b)
		flux_norm = flux/flux_cont
		
		return flux_cont, flux_norm

	def dip_model(abs_line, amplitude, sigma):
		'''
		Returns a dip model and its parameters.
		'''
		cnst = mods.ConstantModel()
		cnst.set_param_hint('c', value=1, vary=True)
		psvt = mods.PseudoVoigtModel()
		psvt.set_param_hint('center', value=abs_line, min=0.8*abs_line, max=1.2*abs_line, vary=True)
		psvt.set_param_hint('amplitude', value=amplitude, vary=True)
		psvt.set_param_hint('sigma', value=sigma, vary=True)
		dpar = psvt.make_params()
		dpar.update(cnst.make_params())
		dmodel = cnst-psvt
		return dmodel, dpar
	
	def fit_dip(lmbd, flux, dmodel, dpar):
		'''
		Completes one iteration of fitting the dip.
		'''
		dfit = dmodel.fit(flux, dpar, x=lmbd)
		dpar = dfit.params
		flux_mdip = dmodel.eval(dpar, x=lmbd)
		mu = dfit.params['center'].value
		return flux_mdip, mu, dpar
	
	def find_center_line(lmbd, flux, ivar, abs_line, window_cont=300):
		'''
		Given a spectrum and an expected absorption line, finds the observed central wavelength of the line.
		'''
		cdx = np.searchsorted(lmbd, (abs_line))
		idxs = np.arange(cdx-window_cont//2, cdx+window_cont//2)
		lmbd_crop = lmbd[idxs]
		flux_crop = flux[idxs]
		ivar_crop = ivar[idxs]

		flux_cont, flux_norm = fit_cont(lmbd_crop, flux_crop, ivar_crop)
		dmodel, dpar = dip_model(abs_line, amplitude=0.5, sigma=np.ptp(lmbd_crop)/4)

		window_iter = 100 # initial window width for dip-fitting
		window_step = 15 # iterative reduction in size
		cdx = len(lmbd_crop)//2 # center of array
		mus = []

		while window_iter > 0:
			idxs = np.arange(cdx-window_iter//2, cdx+window_iter//2)
			lmbd_fcrp = lmbd_crop[idxs] # crop for fitting
			flux_fcrp = flux_crop[idxs]
			flux_mdip, mu, dpar = fit_dip(lmbd_fcrp, flux_fcrp, dmodel, dpar)
			mus.append(mu)
			window_iter -= window_step
		
		return np.mean(mus)
	
	def adjust_redshift(lprp, lobs, lmbd):
		diff = lprp / lobs
		return lmbd * diff

	balmer = 6564.61
	for s in spectra:
		center = find_center_line(s.lmbd, s.flux, s.ivar, balmer)
		print(center)
		s.lmbd = adjust_redshift(balmer, center, s.lmbd)
	
	return spectra

def cont_norm(spectra):
	'''
	Normalizes the temperature contribution to the continuum of the spectrum.
	Modifies `spectrum.flux`.
	'''
	balmer = (np.arange(3, 9))**2
	balmer = 3645.07 * (balmer/(balmer-4))
	crop_wdth = np.arange(100, 140, 5) # 145 comes from 100 + (9-1) * 5

	SpecSpline = namedtuple('SpecSpline', ['spline', 'blue_most', 'slop', 'intr'])

	def crop(lmbd, flux, ivar, crop_strt=8, crop_step=25, crop_wdth=crop_wdth, mean_wdth=2):
		'''
		Given the input wavelengths, finds the mask for cropping out the Balmer lines, including only
		a limited number of sampled points to prevent continuum overfitting.
		'''
		# sample points
		mask = np.zeros(lmbd.shape, dtype=bool)
		idxs = np.arange(crop_strt, len(lmbd), crop_step)
		mask[idxs] = True

		# remove bad points
		mask[np.argwhere(ivar < 0.1)] = False # too much noise
		line_idxs = np.searchsorted(lmbd, balmer)
		for cdx, wdth in zip(line_idxs, crop_wdth):
			lo = max(cdx-wdth, 0)
			hi = min(cdx+wdth, len(lmbd))
			mask[lo:hi] = False
		
		# crop mean
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

	def spline(lmbd, lmbd_crop, flux_crop, ivar_crop):
		'''
		Returns the spline (with blue-most extrapolation) for the spectrum.
		'''
		spline = USpline(lmbd_crop, flux_crop, w=ivar_crop, k=1, s=100)
		return spline
		# blue_most = lmbd_crop[0]

		# # get blue extrapolation
		# frst_derv = spline.derivative()
		# drvs = frst_derv(lmbd)
		# x_4k, x_5k, x_6k = np.searchsorted(lmbd, (4000, 5000, 6000))

		# dr4k = np.mean(drvs[x_4k:x_5k]) # derivative centered on 4500
		# dr5k = np.mean(drvs[x_5k:x_6k]) # derivative centered on 5500
		# lb4k = np.mean(lmbd[x_4k:x_5k]) # exact lambda, roughly 4500
		# lb5k = np.mean(lmbd[x_5k:x_6k]) # exact lambda, roughly 5500
		# scnd_derv = (dr4k - dr5k) / (lb4k - lb5k) # get second derivative between these two points

		# dist = (blue_most - lmbd[0]) / 2 # distance to middle of extrapolated section
		# b_fl, b_sl = spline.derivatives(blue_most) # get flux, slope at blue-most kept point
		# slop = b_sl - scnd_derv * dist
		# intr = b_fl - slop * blue_most
		
		# return SpecSpline(spline, blue_most, slop, intr)
	
	def eval(x, spl):
		'''
		Evaluates spl at x.
		'''
		return spl(x)
		# return np.where(x < spl.blue_most, spl.intr + spl.slop * x, spl.spline(x))
	
	def compute_cont_norm(lmbd, flux, spl):
		'''
		Computes the continuum and normalized flux for the spectrum.
		'''
		cont = eval(lmbd, spl)
		norm = flux / cont
		norm = np.where(norm > 2., 2, norm)
		norm = np.where(norm < -0.5, -0.5, norm)
		return norm
	
	for s in spectra:
		lmbd_crop, flux_crop, ivar_crop = crop(s.lmbd, s.flux, s.ivar)
		spl = spline(s.lmbd, lmbd_crop, flux_crop, ivar_crop)
		s.flux = compute_cont_norm(s.lmbd, s.flux, spl)

	return spectra

def grid_interp(spectra, grid):
	'''
	Interpolates the spectra's x-axes to the specified grid.
	Modifies `spectrum.lmbd` and `spectrum.flux`.
	'''
	for s in spectra:
		f = interp(s.lmbd, s.flux, fill_value='extrapolate', assume_sorted=True)
		s.lmbd = np.copy(grid)
		s.flux = f(grid)

	return spectra

def pipe(spectra, grid):
	spectra = preprocess(spectra)
	spectra = cont_norm(spectra)
	spectra = redshift(spectra)
	spectra = grid_interp(spectra, grid)
	return spectra

if __name__ == '__main__':
	def plot(s, title):
		l = s.lmbd
		f = s.flux
		plt.figure(figsize=(10, 6))
		plt.title('Pipeline - ' + title)
		plt.plot(l, f)
		plt.savefig('../data/pipe_graph_'+title+'.png')
		print(title)
		print(f'lmbd: {l[:10]}, ..., {l[-10:]}')
		print(f'flux: {f[:10]}, ..., {f[-10:]}')
		print('-'*10)

	files = glob.glob('../data/*.fits')
	filename = files[0]
	with fits.open(filename) as f:
		sid = str(f[0].header['PLATEID']) + ' ' + str(f[0].header['MJD']) + ' ' + str(f[0].header['FIBERID'])
		lmbd = 10**f[1].data['loglam']
		flux = f[1].data['flux']
		ivar = f[1].data['ivar']
	
	spectra = [Spectrum(sid, lmbd, flux, ivar)]
	# spectra = preprocess(spectra)
	# plot(spectra[0], 'preprocess')

	# spectra = cont_norm(spectra)
	# plot(spectra[0], 'cont_norm')
	
	# spectra = redshift(spectra)
	# plot(spectra[0], 'redshift')

	# spectra = grid_interp(spectra, np.linspace(3650, 10400, 4000))
	# plot(spectra[0], 'grid_interp')
	
	spectra = pipe(spectra, np.linspace(3650, 10400, 4000))
	plot(spectra[0], 'full_pipe')
