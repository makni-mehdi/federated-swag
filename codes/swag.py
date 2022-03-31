import torch
import numpy as np
import abc
from sklearn.decomposition import TruncatedSVD
# from sklearn.decomposition.pca import _assess_dimension_
from sklearn.utils.extmath import randomized_svd

def flatten(lst):
    tmp = [i.contiguous().view(-1,1) for i in lst]
    return torch.cat(tmp).view(-1)

def set_weights(model, vector, device=None):
    offset = 0
    for param in model.parameters():
        param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()

def _assess_dimension_(spectrum, rank, n_samples, n_features):
    """Compute the likelihood of a rank ``rank`` dataset

    The dataset is assumed to be embedded in gaussian noise of shape(n,
    dimf) having spectrum ``spectrum``.

    Parameters
    ----------
    spectrum : array of shape (n)
        Data spectrum.
    rank : int
        Tested rank value.
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.

    Returns
    -------
    ll : float,
        The log-likelihood

    Notes
    -----
    This implements the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`
    """
    if rank > len(spectrum):
        raise ValueError("The tested rank cannot exceed the rank of the"
                         " dataset")

    pu = -rank * log(2.)
    for i in range(rank):
        pu += (gammaln((n_features - i) / 2.) -
               log(np.pi) * (n_features - i) / 2.)

    pl = np.sum(np.log(spectrum[:rank]))
    pl = -pl * n_samples / 2.

    if rank == n_features:
        pv = 0
        v = 1
    else:
        v = np.sum(spectrum[rank:]) / (n_features - rank)
        pv = -np.log(v) * n_samples * (n_features - rank) / 2.

    m = n_features * rank - rank * (rank + 1.) / 2.
    pp = log(2. * np.pi) * (m + rank + 1.) / 2.

    pa = 0.
    spectrum_ = spectrum.copy()
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, len(spectrum)):
            pa += log((spectrum[i] - spectrum[j]) *
                      (1. / spectrum_[j] - 1. / spectrum_[i])) + log(n_samples)

    ll = pu + pl + pv + pp - pa / 2. - rank * log(n_samples) / 2.

    return ll


class Subspace(torch.nn.Module, metaclass=abc.ABCMeta):
    subclasses = {}

    @classmethod
    def register_subclass(cls, subspace_type):
        def decorator(subclass):
            cls.subclasses[subspace_type] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, subspace_type, **kwargs):
        if subspace_type not in cls.subclasses:
            raise ValueError('Bad subspaces type {}'.format(subspace_type))
        return cls.subclasses[subspace_type](**kwargs)

    def __init__(self):
        super(Subspace, self).__init__()

    @abc.abstractmethod
    def collect_vector(self, vector):
        pass

    @abc.abstractmethod
    def get_space(self):
        pass

@Subspace.register_subclass('covariance')
class CovarianceSpace(Subspace):

    def __init__(self, num_parameters, max_rank=20):
        super(CovarianceSpace, self).__init__()

        self.num_parameters = num_parameters

        self.register_buffer('rank', torch.zeros(1, dtype=torch.long))
        self.register_buffer('cov_mat_sqrt',
                             torch.empty(0, self.num_parameters, dtype=torch.float32))

        self.max_rank = max_rank

    def collect_vector(self, vector):
        if self.rank.item() + 1 > self.max_rank:
            self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :]
        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, vector.view(1, -1)), dim=0)
        self.rank = torch.min(self.rank + 1, torch.as_tensor(self.max_rank)).view(-1)

    def get_space(self):
        return self.cov_mat_sqrt.clone() / (self.cov_mat_sqrt.size(0) - 1) ** 0.5

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        rank = state_dict[prefix + 'rank'].item()
        self.cov_mat_sqrt = self.cov_mat_sqrt.new_empty((rank, self.cov_mat_sqrt.size()[1]))
        super(CovarianceSpace, self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                                           strict, missing_keys, unexpected_keys,
                                                           error_msgs)

    
@Subspace.register_subclass('pca')
class PCASpace(CovarianceSpace):

    def __init__(self, num_parameters, pca_rank=20, max_rank=20):
        super(PCASpace, self).__init__(num_parameters, max_rank=max_rank)

        # better phrasing for this condition?
        assert(pca_rank == 'mle' or isinstance(pca_rank, int))
        if pca_rank != 'mle':
            assert 1 <= pca_rank <= max_rank

        self.pca_rank = pca_rank

    def get_space(self):

        cov_mat_sqrt_np = self.cov_mat_sqrt.clone().numpy()

        # perform PCA on DD'
        cov_mat_sqrt_np /= (max(1, self.rank.item() - 1))**0.5

        if self.pca_rank == 'mle':
            pca_rank = self.rank.item()
        else:
            pca_rank = self.pca_rank

        pca_rank = max(1, min(pca_rank, self.rank.item()))
        pca_decomp = TruncatedSVD(n_components=pca_rank)
        pca_decomp.fit(cov_mat_sqrt_np)

        _, s, Vt = randomized_svd(cov_mat_sqrt_np, n_components=pca_rank, n_iter=5)

        # perform post-selection fitting
        if self.pca_rank == 'mle':
            eigs = s ** 2.0
            ll = np.zeros(len(eigs))
            correction = np.zeros(len(eigs))

            # compute minka's PCA marginal log likelihood and the correction term
            for rank in range(len(eigs)):
                # secondary correction term based on the rank of the matrix + degrees of freedom
                m = cov_mat_sqrt_np.shape[1] * rank - rank * (rank + 1) / 2.
                correction[rank] = 0.5 * m * np.log(cov_mat_sqrt_np.shape[0])
                ll[rank] = _assess_dimension_(spectrum=eigs,
                                              rank=rank,
                                              n_features=min(cov_mat_sqrt_np.shape),
                                              n_samples=max(cov_mat_sqrt_np.shape))
            
            self.ll = ll
            self.corrected_ll = ll - correction
            self.pca_rank = np.nanargmax(self.corrected_ll)
            print('PCA Rank is: ', self.pca_rank)
            return torch.FloatTensor(s[:self.pca_rank, None] * Vt[:self.pca_rank, :])
        else:
            return torch.FloatTensor(s[:, None] * Vt)


class SWAG(torch.nn.Module):

    def __init__(self, base, subspace_type,
                 subspace_kwargs=None, var_clamp=1e-6, *args, **kwargs):
        super(SWAG, self).__init__()

        self.base_model = base(*args, **kwargs)
        self.num_parameters = sum(param.numel() for param in self.base_model.parameters())

        self.register_buffer('mean', torch.zeros(self.num_parameters))
        self.register_buffer('sq_mean', torch.zeros(self.num_parameters))
        self.register_buffer('n_models', torch.zeros(1, dtype=torch.long))

        # Initialize subspace
        if subspace_kwargs is None:
            subspace_kwargs = dict()
        self.subspace = Subspace.create(subspace_type, num_parameters=self.num_parameters,
                                        **subspace_kwargs)

        self.var_clamp = var_clamp

        self.cov_factor = None
        self.model_device = 'cpu'
        
    # dont put subspace on cuda?
    def cuda(self, device=None):
        self.model_device = 'cuda'
        self.base_model.cuda(device=device)

    def to(self, *args, **kwargs):
        self.base_model.to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        self.model_device = device.type
        self.subspace.to(device=torch.device('cpu'), dtype=dtype, non_blocking=non_blocking)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def collect_model(self, base_model, *args, **kwargs):
        # need to refit the space after collecting a new model
        self.cov_factor = None

        w = flatten([param.detach().cpu() for param in base_model.parameters()])
        # first moment
        self.mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.mean.add_(w / (self.n_models.item() + 1.0))

        # second moment
        self.sq_mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.sq_mean.add_(w ** 2 / (self.n_models.item() + 1.0))

        dev_vector = w - self.mean

        self.subspace.collect_vector(dev_vector, *args, **kwargs)
        self.n_models.add_(1)

    def _get_mean_and_variance(self):
        variance = torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp)
        return self.mean, variance

    def fit(self):
        if self.cov_factor is not None:
            return
        self.cov_factor = self.subspace.get_space()

    def set_swa(self):
        set_weights(self.base_model, self.mean, self.model_device)
        
    def diag_sample(self, scale=0.5):
        self.fit()
        mean, variance = self._get_mean_and_variance()
        
        z = variance * torch.randn_like(variance)
        
        z *= scale ** 0.5
        sample = mean + z
        return sample

    def sample(self, scale=0.5, diag_noise=True):
        self.fit()
        mean, variance = self._get_mean_and_variance()

        eps_low_rank = torch.randn(self.cov_factor.size()[0])
        z = self.cov_factor.t() @ eps_low_rank
        if diag_noise:
            z += variance * torch.randn_like(variance)
        z *= scale ** 0.5
        sample = mean + z

        # apply to parameters
        set_weights(self.base_model, sample, self.model_device)
        return sample

    def get_space(self, export_cov_factor=True):
        mean, variance = self._get_mean_and_variance()
        if not export_cov_factor:
            return mean.clone(), variance.clone()
        else:
            self.fit()
            return mean.clone(), variance.clone(), self.cov_factor.clone()
