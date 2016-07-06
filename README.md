#CD-fused-types-problems

## Installation
All of the results of the ipython notebook are based on [this branch](https://github.com/yenchenlin/scikit-learn/tree/cd-fused-types-problems), please clone it and type `sudo make` in the root directory to build scikit-learn.

## Problems

In the ipython notebook, I use modified ElasticNet to fit both `np.float64` and `np.float32` data.

- `np.float64` data 

  CD algorithm will converge at `iter:  2`, and note that `dual_norm_XtA:  0.00402187116681`.
  
- `np.float32` data 

  CD algorithm will not converge until it meets its max_iter constraint, note that at `iter:  2`, `dual_norm_XtA:  0.00414848327637`.
  
In `linear_model/cd_fast.pyx`, we use `if gap < tol` to see whether CD algorithms has converged. 
However, value of `gap` depends on `const`. (See [here](https://github.com/yenchenlin/scikit-learn/blob/cd-fused-types-problems/sklearn/linear_model/cd_fast.pyx#L315-L319))

And since `const = alpha / dual_norm_XtA`, the difference of `dual_norm_XtA` between `np.float32` and `np.float64` data determines whether the algorithm is going to converge or not.

And the reason why there's a difference of `dual_norm_XtA` between two data types is [here](https://github.com/yenchenlin/scikit-learn/blob/cd-fused-types-problems/sklearn/linear_model/cd_fast.pyx#L271-L274), which seems like a precision issues.

## Fitted Modle difference

- `np.float64` data 
```
-----Model-----
coef:  [ -2.42662584e-03  -3.23405311e-04  -2.26606091e-03  -4.09731711e-05
   2.36815579e-04  -2.35184599e-03  -1.97140276e-03   2.65987106e-03
  -2.05523620e-04   2.88947000e-03  -8.23956996e-04   1.03828381e-04
   2.16878424e-03   2.52007819e-04   1.82614790e-03  -1.02433215e-05
  -1.02040495e-03   3.78821835e-04  -7.05928649e-04   1.93929478e-03
  -7.95339808e-04   9.60219426e-05  -2.87422712e-03   3.48095138e-03
  -1.07043123e-03   2.42939601e-03   2.65698850e-03   2.30838855e-03
   1.72572969e-03   7.57900144e-04  -2.70316734e-04   9.47966394e-05
  -2.88102505e-03  -1.05059763e-03  -2.82111843e-04  -8.93081088e-05
  -1.34651727e-03  -2.10912355e-03  -5.36561619e-04  -1.78705391e-03]
dual_gap_:  0.493813246525
eps_:  3.34005184625
n_iter:  3
---------------
```
- `np.float32` data (After max_iter=1000)
```
-----Model-----
coef:  [ -2.42662686e-03  -3.23392800e-04  -2.26604822e-03  -4.09707827e-05
   2.36821521e-04  -2.35185120e-03  -1.97140570e-03   2.65986728e-03
  -2.05539080e-04   2.88946461e-03  -8.23946088e-04   1.03828628e-04
   2.16877926e-03   2.52002385e-04   1.82614138e-03  -1.02523700e-05
  -1.02041254e-03   3.78820696e-04  -7.05922081e-04   1.93929370e-03
  -7.95341504e-04   9.60322286e-05  -2.87423586e-03   3.48094734e-03
  -1.07042806e-03   2.42940127e-03   2.65700766e-03   2.30839103e-03
   1.72573759e-03   7.57896807e-04  -2.70330143e-04   9.47993176e-05
  -2.88102520e-03  -1.05059391e-03  -2.82124878e-04  -8.93148390e-05
  -1.34651491e-03  -2.10913876e-03  -5.36567240e-04  -1.78705074e-03]
dual_gap_:  33.1898269653
eps_:  3.34005188942
n_iter:  1000
---------------
```

We can see that the `dual_gap_` of ElasticNet when fitting `np.float32` data is much larger than `eps_`, which makes it can't meet the convergence condition `if gap < tol`. 

**Note:** `dual_gap_` = `gap`, `eps_` = `tol`.
