# neural-fields-matlab
Some codes for simulating neural field models in MATLAB.

One-dimensional neural field is given by:
$$\dfrac{\partial u(x,t)}{\partial t} = -u(x,t) + \int^{\infty}_{-\infty} w (|x-y|)f(u(y,t)-h){\rm d} y + S(x,t)$$
where  $u(x,t)$ represents the activity at time $t$ of a neuron at field position $x$.


