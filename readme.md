# nada-lite

### use neo.py to generate

Hyperparams:

* `lr1` is learning rate for the `frozen generator`
* `lr2` is learning rate for the `style generator`
* `iteration1` is iteration for the `frozen generator`
* `iteration2` is iteration for the `style generator`
* `dir_lambda` is the weight parameter for the `dir_loss`
* `content_lambda` is the weight parameter for the `content_loss` (not used for now)
* `patch_lambda` is the weight parameter for the `patch_loss`
* `norm_lambda` is the weight parameter for the `norm_loss`
* `gol_lambda` is the weight parameter for the `gol_loss` (not used for now) 


* the `source` and `target` decide the process of generating
* use `content_loss` to train the `frozen generator` first, then continue with the loss function to generate a target pic 

Time:
With `iteration1=100` and `iteration2=100`, the cost of generating will be about 100s on RTX2070-maxq

Network:
`mynetwork` is ours network
`styler` is the origin network
`mynetwork_cmp` todo

train:
`neo` is a training process with our new idea and our model
`neo_cmp_inner` is a training process with the original idea and our model
`neo_cmp_outer` is a training process with styler

count:
a script used to count params and FLOPs

as the used time of both groups is nearly the same(about 240s on rtx-2070 maxq), we could assume that the new model is much faster than the control group
