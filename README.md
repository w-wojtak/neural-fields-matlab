# neural-fields-matlab
Some codes for simulating neural field models in MATLAB.

One-dimensional neural field is given by:
$$\dfrac{\partial u(x,t)}{\partial t} = -u(x,t) + \int_{-\Omega} w (|x-y|)f(u(y,t)-h){\rm d} y + I(x,t)$$
where  $u(x,t)$ represents the activity at time $t$ of a neuron at field position $x$.


