import numpy as np
import matplotlib.pyplot as plt
from analyse.acf import gen_mc_acf

def return_params(dataset, snr,tsh=4):
	if dataset == 'reg':
		mu =  np.array([0.0, 1.])
		s =  np.array([1., 1.])/snr
		pi = np.array([0.6, 0.4])
		transition = np.array([
			[0.98, 0.02],
			[0.03, 0.97]])

		return mu, s, pi, transition

	elif dataset == 'dynamic2':
		mu =  np.array([0.0, 1., 0., 1.])
		s =  np.array([1., 1., 1., 1.])/snr
		pi = np.array([0.2, 0.4/3, 0.6, 0.4])

		pi /= pi.sum()

		transition = np.array([
			[0.94, 0.06, 0.0005*tsh, 0.0005*tsh],
			[0.09, 0.91, 0.0005*tsh, 0.0005*tsh],
			[0.0005*tsh/3, 0.0005*tsh/3, 0.98, 0.02],
			[0.0005*tsh/3, 0.0005*tsh/3, 0.03, 0.97]])

		for i in range(transition.shape[0]):
			transition[i] /= transition[i].sum()

		return mu, s, pi, transition

	elif dataset == 'dynamic3':
		mu =  np.array([0.0, 0.75, 1., 0.0, 1.])
		s =  np.array([1., 1., 1., 1., 1.])/snr
		pi = np.array([0.2, 0.5/3, 0.4/3, 0.6, 0.4])

		pi /= pi.sum()

		transition = np.array([
			[0.980, 0.010, 0.010, 0.008/6, 0.008/6],
			[0.0125, 0.975, 0.0125, 0.008/6, 0.008/6],
			[0.015, 0.015, 0.970, 0.008/6, 0.008/6],
			[0.008/18, 0.008/18, 0.008/18, 0.98, 0.02],
			[0.008/18, 0.008/18, 0.008/18, 0.03, 0.97]])

		for i in range(transition.shape[0]):
			transition[i] /= transition[i].sum()

		return mu, s, pi, transition

	elif dataset == 'static2':
		mu =  np.array([0.0, 1.])
		s =  np.array([1., 1.])/snr
		pi = np.array([0.6, 0.4])

		transition1 = np.array([
			[0.94, 0.06],
			[0.09, 0.91]])

		transition2 = np.array([
			[0.98, 0.02],
			[0.03, 0.97]])

		return mu, s, pi, transition1, transition2

	elif dataset == 'static3':
		mu1 =  np.array([0.0, 0.75, 1.])
		s1 =  np.array([1., 1., 1.])/snr
		pi1 = np.array([0.6, 0.5, 0.4])

		transition1 = np.array([[0.980, 0.010, 0.010],
								[0.0125, 0.975, 0.0125],
								[0.015, 0.015, 0.970]])

		mu2 =  np.array([0.0, 1.])
		s2 =  np.array([1., 1.])/snr
		pi2 = np.array([0.6, 0.4])

		transition2 = np.array([[0.98, 0.02],
								[0.03, 0.97]])

		return mu1, s1, pi1, transition1, mu2, s2, pi2, transition2


def return_acf(dataset, t_max, snr, prop=None, tsh=4):
	if dataset in ['static2', 'static3']:
		if dataset == 'static2':
			mu, s, pi, transition1, transition2 = return_params(dataset,snr)
			t, acf1 = gen_mc_acf(1, t_max, transition1, mu, s**2, pi )
			t, acf2 = gen_mc_acf(1, t_max, transition2, mu, s**2, pi )
		else:
			mu1, s1, pi1, transition1, mu2, s2, pi2, transition2 = return_params(dataset,snr)
			t, acf1 = gen_mc_acf(1, t_max, transition1, mu1, s1**2, pi1 )
			t, acf2 = gen_mc_acf(1, t_max, transition2, mu2, s2**2, pi2 )

		acf = (prop[0]*acf1 + prop[1]*acf2)/prop.sum()

	else:
		mu, s, pi, transition = return_params(dataset,snr,tsh)

		t, acf = gen_mc_acf(1, t_max, transition, mu, s**2, pi )

	return t, acf

def plot_acf(t, acfs, dataset, acf_type, model, snr, pb='NoPB', prop=None, ylims = None):
	t_true, true_acf = return_acf(dataset, len(t), snr, prop)


	fig = plt.figure(figsize=(4,3.5))
	for E_y0yt_res in acfs:
		plt.plot(t, E_y0yt_res, lw = 1, alpha = 0.1)


	plt.plot(t, true_acf, 'r--', lw = 1)
	if ylims is not None:
		plt.ylim(ylims)

	figname = dataset + '_' + model + '_' + acf_type + '_' + str(snr) + '_' + pb

	plt.show()
	#fig.savefig('./figures/acf_{}.pdf'.format(figname))
	#fig.savefig('./figures/acf_{}.png'.format(figname),dpi=300)

def plot_acf_residuals(t, acfs, dataset, acf_type, model, snr, pb='NoPB', prop=None,xlims = None, ylims = None):
	t_true, true_acf = return_acf(dataset, len(t), snr, prop)

	residual_Es = true_acf - np.array(acfs)
	mean_Es = np.mean(residual_Es, axis = 0)
	percentile_975 = np.percentile(residual_Es, 97.5, axis = 0)
	percentile_025 = np.percentile(residual_Es, 2.5, axis = 0)

	fig = plt.figure(figsize=(4,3.5))
	for i in range(residual_Es.shape[0]):
		plt.plot(t, residual_Es[i], color='grey', alpha = 0.05)


	plt.plot(t, mean_Es, 'k')
	plt.fill_between(t, percentile_975, percentile_025, color = "#1F77B419", edgecolor='#165582ff', ls = '--')

	if xlims is not None:
		plt.xlim(xlims)
	if ylims is not None:
		plt.ylim(ylims)

	figname = dataset + '_' + model + '_' + acf_type + '_' + str(snr) + '_' + pb

	plt.show()
	#fig.savefig('./figures/acf_residuals_{}.pdf'.format(figname))
	#fig.savefig('./figures/acf_residuals_{}.png'.format(figname),dpi=300)
