# neural-fields-matlab


The first model considered here is the neural field model, first proposed and analyzed by Amari [[1]](#1), governed by the following nonlinear integro-differential equation on a one-dimensional, spatially extended domain $\Omega$

$$\dfrac{\partial u(x,t)}{\partial t} = -u(x,t) + \int_{\Omega} w (|x-y|)f(u(y,t)-\theta){\rm d} y + I(x,t)$$

where  $u(x,t)$ represents the activity at time $t$ of a neuron at field position $x$.

The nonlinearity $f$ denotes the firing rate function, often taken as a smooth sigmoidal function with steepness parameter $\beta$

$f(x) = \dfrac{1}{1+e^{-\beta(x-\theta)}},$

which for large values of $\beta$ approximates the Heaviside step function.

The term $w$ is the distance-dependent coupling function, 

The first one is a coupling function with constant lateral inhibition

$w_ {lat}(x) = A_{lat}e^{\left(-x^{2}/2\sigma^{2}_ {lat}\right)} - g_{lat}$

where $ A_ {lat}> 0 $ and $ \sigma_ {lat} > 0$  and $g_ {lat} > 0$.

The second example is the Mexican hat function given by the difference of two Gaussians

$w_ {mex}(x) = A_{ex}e^{\left(-x^{2}/2\sigma^{2}_ {ex} \right)} - A_{in}e^{\left(-x^{2}/2\sigma^{2}_ {in}\right)} - g_{in}$,

where $ A_{ex}  > A_{in} > 0 $ and $ \sigma_{in} > \sigma_{ex} > 0$  and $g_{in} > 0$.

The third example is the oscillatory connectivity function

$w_ {osc}(x) = \exp(-b|x|) (b \sin |x| + \cos (x) )$

where $b>0$ controls the rate at which the oscillations decay with distance.

The second model is the two field model from [[2]](#2)
$$\dfrac{\partial u(x,t)}{\partial t} = -u(x,t) + v(x,t) + \int_{\Omega} w (|x-y|)f(u(y,t)-\theta){\rm d} y + I(x,t)$$
$$\dfrac{\partial v(x,t)}{\partial t} = -v(x,t) + u(x,t) - \int_{\Omega} w (|x-y|)f(u(y,t)-\theta){\rm d}. y$$


## References
<a id="1">[1]</a> 
Amari, S. I. (1977). Dynamics of pattern formation in lateral-inhibition type neural fields. Biological cybernetics, 27(2), 77-87.

<a id="2">[2]</a> 
Wojtak, W., Coombes, S., Avitabile, D., Bicho, E., & Erlhagen, W. (2021). A dynamic neural field model of continuous input integration. Biological Cybernetics, 115(5), 451-471.

