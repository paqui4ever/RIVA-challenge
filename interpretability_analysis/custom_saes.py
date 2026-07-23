# Copied from the overcomplete library (https://github.com/KempnerInstitute/overcomplete)

"""
Module for Relaxed Archetypal SAE implementations.
For the implementation of the Relaxed Archetypal Dictionary, see archetypal_dictionary.py.
"""

import torch
from torch import nn

from overcomplete.sae.topk_sae import TopKSAE
from overcomplete.sae.jump_sae import JumpSAE
from overcomplete.sae.archetypal_dictionary import RelaxedArchetypalDictionary


class RATopKSAE(TopKSAE):
    """
    Relaxed Archetypal TopK SAE.

    This class implements a TopK SAE that utilizes a Relaxed Archetypal Dictionary.
    The dictionary atoms are initialized and constrained to be convex combinations
    of data points.

    For more information, see:
        - "Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in
          Large Vision Models" by T. Fel et al., ICML 2025 (https://arxiv.org/abs/2502.12892).

    Parameters
    ----------
    input_shape : int
        Dimensionality of the input data (excluding the batch dimension).
    nb_concepts : int
        Number of dictionary atoms (concepts).
    points : torch.Tensor
        The data points used to initialize/define the archetypes.
        Shape should be (num_points, input_shape).
    top_k : int
        Number of top activations to keep in the latent representation.
        By default, 10% sparsity is used.
    delta : float, optional
        Delta parameter for the archetypal dictionary, by default 1.0.
    use_multiplier : bool, optional
        Whether to use a learnable multiplier that parametrize the ball (e.g. if this parameter
        is 3 then the dictionary atoms are all on the ball of radius 3). By default True.
    **kwargs : dict, optional
        Additional arguments passed to the parent TopKSAE (e.g., encoder_module, device).
    """

    def __init__(self, input_shape, nb_concepts, points, top_k=None, delta=1.0, use_multiplier=True, **kwargs):
        assert isinstance(input_shape, int), "RATopKSAE input_shape must be an integer."

        super().__init__(input_shape=input_shape, nb_concepts=nb_concepts,
                         top_k=top_k, **kwargs)

        # enforce archetypal dictionary after the init of the parent class
        self.dictionary = RelaxedArchetypalDictionary(
            in_dimensions=input_shape,
            nb_concepts=nb_concepts,
            points=points,
            delta=delta,
            use_multiplier=use_multiplier,
            device=self.device
        )


class RAJumpSAE(JumpSAE):
    """
    Relaxed Archetypal Jump SAE.

    This class implements a Jump SAE that utilizes a Relaxed Archetypal Dictionary.
    The dictionary atoms are initialized and constrained to be convex combinations
    of data points.

    For more information, see:
        - "Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in
          Large Vision Models" by T. Fel et al., ICML 2025 (https://arxiv.org/abs/2502.12892).

    Parameters
    ----------
    input_shape : int
        Dimensionality of the input data (excluding the batch dimension).
    nb_concepts : int
        Number of dictionary atoms (concepts).
    points : torch.Tensor
        The data points used to initialize/define the archetypes.
        Shape should be (num_points, input_shape).
    bandwidth : float, optional
        Bandwidth parameter for the Jump SAE kernel, by default 1e-3.
    delta : float, optional
        Delta parameter for the archetypal dictionary, by default 1.0.
    use_multiplier : bool, optional
        Whether to use a learnable multiplier that parametrize the ball (e.g. if this parameter
        is 3 then the dictionary atoms are all on the ball of radius 3). By default True.
    **kwargs : dict, optional
        Additional arguments passed to the parent JumpSAE (e.g., encoder_module, device).
    """

    def __init__(self, input_shape, nb_concepts, points, bandwidth=1e-3, delta=1.0, use_multiplier=True, **kwargs):
        assert isinstance(input_shape, int), "RAJumpSAE input_shape must be an integer."

        super().__init__(input_shape=input_shape, nb_concepts=nb_concepts,
                         bandwidth=bandwidth, **kwargs)

        # enforce archetypal dictionary after the init of the parent class
        self.dictionary = RelaxedArchetypalDictionary(
            in_dimensions=input_shape,
            nb_concepts=nb_concepts,
            points=points,
            delta=delta,
            use_multiplier=use_multiplier,
            device=self.device
        )


"""
Module for Matching Pursuit Sparse Autoencoder (MpSAE).
"""

import torch
from torch import nn

from overcomplete.sae import SAE


class MpSAE(SAE):
    """
    Matching Pursuit Sparse Autoencoder (MpSAE).

    This autoencoder uses a greedy Matching Pursuit strategy to obtain sparse
    codes. Specifically, at each iteration the dictionary atom most correlated
    with the current residual is chosen and its contribution is subtracted
    from the residual. This process is repeated k times. An optional dropout
    can be applied on the dictionary elements at each iteration.
    Warning: for this SAE, the encoding is returning (1) the residual and (2) the
    codes -- as the pre_codes are just the input.

    Parameters
    ----------
    input_shape : int or tuple of int
        Dimensionality of the input data (excluding the batch dimension).
    nb_concepts : int
        Number of latent dimensions (components) of the autoencoder.
    k : int, optional
        The number of matching pursuit iterations to perform (must be > 0).
    dropout : float, optional
        Probability of dropping a dictionary element at each iteration
        (range: 0.0 - 1.0). If None, no dropout is applied.
    encoder_module : nn.Module or str, optional
        Custom encoder module (or its registered name). If None, a default encoder is used.
    dictionary_params : dict, optional
        Parameters that will be passed to the dictionary layer.
        See DictionaryLayer for more details.
    device : str, optional
        Device on which to run the model (default is 'cpu').
    """

    def __init__(self, input_shape, nb_concepts, k=1, dropout=None,
                 encoder_module="identity", dictionary_params=None, device='cpu'):
        # input shape must be int or length-1 tuple
        assert isinstance(input_shape, int) or len(input_shape) == 1, \
            "MpSAE Doesn't handle 3d or 4d data format."
        if isinstance(k, int):
            assert k > 0, "k must be a positive integer."

        super().__init__(input_shape, nb_concepts,
                         encoder_module, dictionary_params, device)
        self.k = k
        self.dropout = dropout

    def encode(self, x):
        """
        Encode input data with a greedy Matching Pursuit approach.

        The dictionary (W) is optionally subjected to dropout. Then, at each
        of k iterations:
            1) Compute the correlation (dot product) between the residual and each
               dictionary atom.
            2) Identify the maximum correlation and corresponding dictionary atom index.
            3) Update the codes by adding that contribution.
            4) Subtract the chosen atom (scaled by the correlation) from the residual.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        residual : torch.Tensor
            The residual after k Matching Pursuit iterations.
        codes : torch.Tensor
            The final sparse codes obtained after k Matching Pursuit iterations.
        """
        W = self.get_dictionary()

        if self.dropout is not None:
            # dropout directly on the weights (dictionary) atoms
            drop_w = torch.bernoulli((1.0 - self.dropout) *
                                     torch.ones(self.nb_concepts, 1, device=self.device))
            W = W * drop_w

        codes = torch.zeros(x.shape[0], self.nb_concepts, device=self.device)
        residual = x.clone()

        # greedy selection of dictionary atoms
        for _ in range(self.k):
            z = residual @ W.T  # pre_codes as projection of the current residual
            val, idx = torch.max(z, dim=1)

            # add top concept to the current codes
            to_add = torch.nn.functional.one_hot(idx, num_classes=self.nb_concepts).float()
            to_add = to_add * val.unsqueeze(1)

            # accumulate contribution and update residual
            codes = codes + to_add
            residual = residual - to_add @ W

        return residual, codes

    def train(self, mode=True):
        """
        Hook called when switching between training and evaluation mode.
        We use it to ensure no dropout is applied during evaluation.

        Parameters
        ----------
        mode : bool, optional
            Whether to set the model to training mode or not, by default True.
        """
        if not mode:
            self.dropout = None

        return super().train(mode)


"""
Module for Orthogonal Matching Pursuit Sparse Autoencoder (OMPSAE).
"""

import torch
from overcomplete.sae import SAE
from opt_utils import batched_matrix_nnls


class OMPSAE(SAE):
    """
    Orthogonal Matching Pursuit Sparse Autoencoder (OMPSAE).

    This autoencoder uses an Orthogonal Matching Pursuit strategy to find sparse
    codes. At each iteration, the atom most correlated with the current residual
    is selected, and the full NNLS problem is solved on the selected atoms to update
    the codes. Optionally, dictionary elements can be randomly dropped at each iteration.
    Warning: for this SAE, the encoding is returning (1) the residual and (2) the
    codes -- as the pre_codes are just the input.

    Parameters
    ----------
    input_shape : int or tuple of int
        Dimensionality of the input data (excluding batch dimension).
    nb_concepts : int
        Number of latent components (atoms) in the dictionary.
    k : int, optional
        Default number of pursuit iterations (must be > 0).
    dropout : float, optional
        Dropout rate applied to dictionary atoms (range [0.0, 1.0]).
    encoder_module : str or nn.Module, optional
        Encoder module or name of registered encoder.
    dictionary_params : dict, optional
        Parameters passed to the dictionary layer.
    device : str, optional
        Device to run the model on (default is 'cpu').
    max_iter : int, optional
        Default number of NNLS iterations (default: 10).
    """

    def __init__(self, input_shape, nb_concepts, k=1, dropout=None,
                 encoder_module="identity", dictionary_params=None, device='cpu',
                 max_iter=10):

        assert isinstance(input_shape, int) or len(input_shape) == 1, \
            "OMPSAE doesn't support 3D or 4D input format."
        assert isinstance(k, int) and k > 0, "k must be a positive integer."
        if dropout is not None:
            assert 0.0 <= dropout <= 1.0, "Dropout must be in range [0, 1]."
        assert isinstance(max_iter, int) and max_iter > 0, "max_iter must be a positive integer."

        super().__init__(input_shape, nb_concepts, encoder_module, dictionary_params, device)
        self.k = k
        self.dropout = dropout
        self.max_iter = max_iter

    def encode(self, x, k=None, max_iter=None):
        """
        Encode input using Orthogonal Matching Pursuit.

        At each iteration of the pursuit, (1) select the atom most correlated with the residual,
        (2) solve NNLS over all selected atoms, and (3) update codes and residual.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        k : int, optional
            Override the number of pursuit iterations.
        max_iter : int, optional
            Override the number of NNLS iterations.

        Returns
        -------
        residual : torch.Tensor
            Final residual after k iterations.
        codes : torch.Tensor
            Sparse codes of shape (batch_size, nb_concepts).
        """
        k = k if k is not None else self.k
        max_iter = max_iter if max_iter is not None else self.max_iter

        assert isinstance(k, int) and k > 0, "k must be a positive integer."
        assert isinstance(max_iter, int) and max_iter > 0, "max_iter must be a positive integer."

        W = self.get_dictionary()

        if self.dropout is not None:
            drop_mask = torch.bernoulli(
                (1.0 - self.dropout) * torch.ones(W.shape[0], device=self.device)
            )
            W = W * drop_mask.unsqueeze(1)

        batch_size = x.shape[0]
        codes = torch.zeros(batch_size, self.nb_concepts, device=self.device)
        residual = x.clone()
        selected_atoms = None

        with torch.no_grad():
            for _ in range(k):
                z = residual @ W.T

                if selected_atoms is not None:
                    z.scatter_(dim=1, index=selected_atoms, value=-torch.inf)

                _, idx = torch.topk(z, k=1, dim=1)
                selected_atoms = idx if selected_atoms is None else torch.cat([selected_atoms, idx], dim=1)

                W_sel = W[selected_atoms]
                Z_init = torch.gather(codes, 1, selected_atoms)

                codes_sel = batched_matrix_nnls(
                    W_sel,
                    x,
                    max_iter=max_iter,
                    tol=1e-5,
                    Z_init=Z_init
                )

                codes.scatter_(dim=1, index=selected_atoms, src=codes_sel)
                residual = x - codes @ W

        return residual, codes

    def train(self, mode=True):
        """
        Hook called when switching between training and evaluation mode.
        We use it to ensure no dropout is applied during evaluation.

        Parameters
        ----------
        mode : bool, optional
            Whether to set the model to training mode or not, by default True.
        """
        if not mode:
            self.dropout = None

        return super().train(mode)


