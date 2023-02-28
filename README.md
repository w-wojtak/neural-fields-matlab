# neural-fields-matlab
Some codes for simulating neural field models in MATLAB.

The dynamics of the neural field model proposed and analyzed by Amari [[1]](#1) is governed by the following nonlinear integro-differential equation on a one-dimensional, spatially extended domain $\Omega$

$$\dfrac{\partial u(x,t)}{\partial t} = -u(x,t) + \int_{\Omega} w (|x-y|)f(u(y,t)-\theta){\rm d} y + I(x,t)$$
where  $u(x,t)$ represents the activity at time $t$ of a neuron at field position $x$.

The nonlinearity $f$ denotes the firing rate function with threshold $\theta$ and $w$ is the distance-dependent coupling function.



## References
<a id="1">[1]</a> 
Amari, S. I. (1977). Dynamics of pattern formation in lateral-inhibition type neural fields. Biological cybernetics, 27(2), 77-87.

