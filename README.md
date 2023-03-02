# neural-fields-matlab

The dynamics of the neural field model proposed and analyzed by Amari [[1]](#1) is governed by the following nonlinear integro-differential equation on a one-dimensional, spatially extended domain $\Omega$

$$\dfrac{\partial u(x,t)}{\partial t} = -u(x,t) + \int_{\Omega} w (|x-y|)f(u(y,t)-\theta){\rm d} y + I(x,t)$$
where  $u(x,t)$ represents the activity at time $t$ of a neuron at field position $x$.

The nonlinearity $f$ denotes the firing rate function with threshold $\theta$ and $w$ is the distance-dependent coupling function.

The second model is the two field model from [[2]](#2)
$$\dfrac{\partial u(x,t)}{\partial t} = -u(x,t) + v(x,t) + \int_{\Omega} w (|x-y|)f(u(y,t)-\theta){\rm d} y + I(x,t)$$
$$\dfrac{\partial v(x,t)}{\partial t} = -v(x,t) + u(x,t) - \int_{\Omega} w (|x-y|)f(u(y,t)-\theta){\rm d} y$$


## References
<a id="1">[1]</a> 
Amari, S. I. (1977). Dynamics of pattern formation in lateral-inhibition type neural fields. Biological cybernetics, 27(2), 77-87.

<a id="2">[2]</a> 
Wojtak, W., Coombes, S., Avitabile, D., Bicho, E., & Erlhagen, W. (2021). A dynamic neural field model of continuous input integration. Biological Cybernetics, 115(5), 451-471.

