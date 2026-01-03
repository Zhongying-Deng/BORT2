import torch

try:
    import intel_extension_for_pytorch as ipex
except:
    pass

__all__ = ['Hypergrad']


class Hypergrad:
    """Implicit differentiation for auxiliary parameters.
    This implementation follows the Algs. in "Optimizing Millions of Hyperparameters by Implicit Differentiation"
    (https://arxiv.org/pdf/1911.02590.pdf), with small differences.

    """

    def __init__(self, learning_rate=.1, truncate_iter=3):
        self.learning_rate = learning_rate
        self.truncate_iter = truncate_iter

    def grad(self, loss_val, loss_train, aux_params, params):
        """Calculates the gradients w.r.t \phi dloss_aux/dphi, see paper for details

        :param loss_val:
        :param loss_train:
        :param aux_params:
        :param params:
        :return:
        """
        dloss_val_dparams = torch.autograd.grad(
            loss_val,
            params,
            retain_graph=True,
            allow_unused=True
        )
        dloss_val_dp = []
        use_cuda = False

        self.device = loss_val.device
                    
        for ind, v in enumerate(dloss_val_dparams):
            if v is None:
                #v = torch.zeros_like(params[ind]) if use_cuda else torch.zeros_like(params[ind]).cuda()
                v = torch.zeros_like(params[ind]).to(self.device)
            else:
                if v.is_cuda:
                    use_cuda = True
            dloss_val_dp.append(v)


        dloss_train_dparams = torch.autograd.grad(
                loss_train,
                params,
                allow_unused=True,
                create_graph=True,
        )

        #v2 = self._approx_inverse_hvp(dloss_val_dparams, dloss_train_dparams, params)
        v2 = self._approx_inverse_hvp(dloss_val_dp, dloss_train_dparams, params)

        v3 = torch.autograd.grad(
            dloss_train_dparams,
            aux_params,
            grad_outputs=v2,
            allow_unused=True
        )
        # note we omit dL_v/d_lambda since it is zero in our settings
        #for ind, param in enumerate(v3):
        #    if param is None:
        #        print(ind, param)
        return list(-g for g in v3)

    def _approx_inverse_hvp(self, dloss_val_dparams, dloss_train_dparams, params):
        """

        :param dloss_val_dparams: dL_val/dW
        :param dloss_train_dparams: dL_train/dW
        :param params: weights W
        :return: dl_val/dW * dW/dphi
        """
        p = v = dloss_val_dparams
        for ind, val in enumerate(dloss_val_dparams):
            if val is None:
                print('dval {}'.format(ind))
        dloss_train_dp = []
        for ind, val in enumerate(dloss_train_dparams):
            if val is None:
                #val = torch.zeros_like(params[ind]) if False else torch.zeros_like(params[ind]).cuda()
                val = torch.zeros_like(params[ind]).to(self.device)
                dloss_train_dp.append(val)
                print('dtrain {}, param shape {}'.format(ind, params[ind].shape))
        for ind, val in enumerate(params):
            if val is None:
                print('param {}'.format(ind))
        for _ in range(self.truncate_iter):
            grad = torch.autograd.grad(
                    dloss_train_dparams,
                    params,
                    grad_outputs=v,
                    retain_graph=True,
                    allow_unused=True
                )

            grad = [g * self.learning_rate for g in grad]  # scale: this a is key for convergence

            v = [curr_v - curr_g for (curr_v, curr_g) in zip(v, grad)]
            # note: different than the pseudo code in the paper
            p = [curr_p + curr_v for (curr_p, curr_v) in zip(p, v)]

        return list(pp for pp in p)
