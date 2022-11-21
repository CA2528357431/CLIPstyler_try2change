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



Path:

pic1/: the result of "training `frozen generator`"

pic2/: the result of "training `style generator`"

pic3/: the result of "control group"



Details:

* the `source` and `target` decide the process of generating
* use `content_loss` to train the `frozen generator` first, then continue with the loss function to generate a target pic 

Time:
With `iteration1=250` and `iteration2=250`, the cost of generating will be about 240s on RTX2070-maxq



Further, `neo_cmp` and `mynetwork_cmp` are "control group", which are used to compare to the new one.



for example, 



photo->fire

<img src="file:///E:/stylegan_nada/ori0.jpg" title="" alt="" width="188">



train frozen generator(50 iterations per picture)

![](C:\Users\win10\Downloads\png%20(2).png)

train style generator(10 iterations per picture)

![](C:\Users\win10\Downloads\png.png)



control group(10 iterations per picture)

![](C:\Users\win10\Downloads\png%20(1).png)



as the used time of both groups is nearly the same(about 240s on rtx-2070 maxq), we could assume that the new model is much faster than the control group
