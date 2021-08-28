---
layout: post
comments: true
title: "Dsp Lab Pointers ⤴ "
date: 2021-08-27 12:00:00
---

> Digital Signal Processing lab guide
<!--more-->

{:class="table-of-content"}
* TOC
{:toc}

# Brief note
This is a minimal question by question guide to the Digial Signal Processing(DSP) lab. Each approach provided is for an existing question from the lab manual; so please check just to make sure I'm not making things up. I may or may not be sober while writing this.

You can access the said lab manual from [ here ]({{'assets/temp/ETI_2505_Labwork.pdf' | relative_url }}) in case you have no idea what manual we are talking about; which is allowed by the way.

### What this guide doesn't provide:
- Answers to the analysis questions. Just run the script, observe the output and analyse the output your own way -- it should be fairly easy.
    By analysis questions, we mean questions like:

    > "What happens with your plots when increasing the frequency?"

    > Does the plot look beautiful? Motivate your answer

_Oh, and this shit is written in Octave. If you fell in love with Matlab, it should still work fine. In case of error though, just point it out -- we'll resolve it so you can continue enjoying your relationship with Matlab._


# Lab1
## Exercise 1

$$
x(n) = sin(\omega n) , n = 1, \dots,  40
$$

Take $\omega = 0.8$
```matlab
    n = 1:40 % 40 samples
    x_n = sin(w *n) 

    % Plot stem of each x;
    stem(sin(x4));
    xlabel('Time');
    title('f = Your w');

```

Part b

Using $\triangle T = 0.1$. 
```matlab
 T = .1
 n = 1:T: 40;
 w = .1 % Omega
  x_n = sin(w *n)

 % Plot stem of each x;
 stem(sin(x4));
 xlabel('Time');
 title('f = 1/.1');
```
Repeat both parts for the rest of $\omega = 0.1, 0.2, 0.4, \text{ and } 0.8.$


## Exercise 2
$$
X(t) = A \cos(2\pi f_ct +\theta), t \in [t_{ start }, t_{ end }]
$$

```matlab
A = 5, f= 10, phi = pi /3;
s = 0;
et = 100;

%dt = 10^-2
dt = 0.01;
t = s: dt: et;

x = A * cos(2 * pi * f * t + phi)

clf;
stem(t, x)

```


## Exercise 3
$\delta(n - 15)$ and $u(n -10)$

Remember:
$$
\delta(n) = 
\begin{cases}
1, n = 0\\
0, \text{ otherwise }
\end{cases}
\\
u(n) = 
\begin{cases}
1, n >= 0\\
0, \text{ otherwise }
\end{cases}
\\
$$

Not that anyone cares to know any longer :]
```matlab

n = 1: 30;

delta_n = n ==15;
ustep_n = n >= 10;
```

$x_1 = u(n - 10) - u(n -11)$
```matlab
% Plot d(n)
stem(n, delta_n);
title('Unit delta_n - delta(n - 15)');
%xlabel('Time (s)');


% Plot u(n)
stem(n, ustep_n);
xlabel('Time');
ylabel('Your y label');
title("I'm the unit step yo!");


% u(n - 11)
ustep_n1 = n >= 11;

x1 = ustep_n - ustep_n1;

clf;
stem(n, x1);
xlabel('Time');
ylabel('Your y label');
title("I'm the unit step yo!");
```

$x_2(n) = u(n - 10) - u(n - 15)$
```matlab


% x2(n) = u(n - 10) - u(n - 15)

% u2 = u(n - 15)
ustep_n2 = n >= 15

x2 = ustep_n - ustep_n2
clf;
stem(n, x2, marker='.');
xlabel('Time');
ylabel('Your y label');
title("I'll say it again, I'm the unit step");


```

## Exercise 4
$$x(n) = e^{j\omega}, n = 1,\dots, 40$$

```matlab
n = 1: 40;
w = .2, j = sqrt(-1);

x_n = e.^(j*w*n);

% Extract real and imaginary parts
x = real(x_n);
y = imag(x_n);



clf;
stem(n, x);


% Plot the real and imaginary parts
title('Real component');
ylabel('A');

stem(n, y);

title('Imaginary component');
ylabel('A');
xlabel('Time');
```

# Lab 2 
## 2.1 LTI systems
### Exercise 1
$$
 h(n) = 0.02, where
 \begin{cases}    0 \le n \le 49\\
                   0, otherwise
\end{cases}
$$

$x(n) = 3+ sin(0.4n), 0 \lt n \lt 299$

```matlab
n = 0:299; % 300 samples 

h = n
h(1:50) = .02; h(51:end) = 0 % 0.02 where 0 < n < 49

```

Create another signal:
```matlab
% x(n) =3 + sin (0.4n), 0 < n < 99
x = 3 + sin(0.4 *n)

% Convolve the signals
y = conv(h, x)
```

Answer question in manual


### Exercise 2

```matlab
n = 0:599
xi = randn(1, 600)

x = sin(.01 * n) + xi

y = conv(h, x)

```

### Exercise 3

```matlab
n = 0:1: 9

s = .1 * n

vec0 = s


vec1 = conv(vec0, vec0)
vec2 = conv(vec1, vec0)
vec3 = conv(vec2, vec0)
vec4 = conv(vec3, vec0)

stem(vec1) % Repeat for all other vectors
```

## 2.2 DTFT
Create a DTFT function file. Call it `dtft.m` and save it. It should have the following DTFT function:
```matlab
% Shamelessly copied from https://www.mathworks.com/matlabcentral/answers/469374-dtft-possible-on-matlab

function [X] = dtft(x,n,w)
    % Computes Discrete-time Fourier Transform
    % [X] = dtft(x,n,w)
    %   X = DTFT values computed at w frequencies
    %   x = finite duration sequence over n
    %   n = sample position vector
    %   w = frequency location vector
    X = exp(-1i*w'*n)  * x.';
    % X = x*exp(-j*n'*w);
end
```
We shall use this function throughout the rest of the exercise

### Exercise 1

```matlab
% Use linspace to get the required frequency

number_of_samples = 500 % Modify this value to your own

fq = linspace(-1.5, 1.5, number_of_samples)
x = -10: .1: 20
x = zeros(1, number_of_samples)
n = x

x(x >= - 5 & x <=5) = 1
x(x != 1) = 0

w = 2 * pi * fq;

X = dtft(x_n, n, w)

% Plot DTFT
plot(w, X)
% Add title, xlabel, blah blah blah

```
### Exercise 2
$$
x(n) =  
\begin{cases}
2 \pi f_0 n, n = -30 \lt n \lt 30 \\
0, \text{ otherwise }
\end{cases}
$$

```matlab
  start = 100 ; % Select a start and end value of your choice
  end_ = 800 ; % Modify these two !!!!!!!!!!!!!
```

Start with $f_0 = 0.05$

```matlab
f0 = 0.05

n = start: end_
x = zeros(length(n))

% Loop around inserting cos where the 
% condition for x is met

for idx = 1: length(x)
    element = n(idx)

    % Condition met
    % Tumekapata!!
    if element > = -30 & element <=30
        x(idx) = cos(2*pi* f0*element)
    end

    % Kametupitaaa
end
```

DTFTs of $X(f) \text{ vs } f \in [-0.5, 0.5]$. It's said $f$ has $0.01$ intervals. We are also told to use `linspace`. To do this, we need to know how many points of $f$ we have between $-0.5$ and $0.5$ at a $0.01$ interval.


How many are they? Chukua calc

.

.

.

.
.

.
.
.


Bado huna calc

Let me leave you some space to pick your calc

```matlab




% SPACE YA KUCHUKUA CALC







```

Anyway, we need $301$ points. So next step is finding the DTFT of $x(n)$

```matlab
    f = linspace(-1.5, 1.5, 301) % 300 points for 0.01 intervals

    w = 2 * pi * f;

    X = dtft(x, n, w)
    plot(w, X)
```

Now repeat this procedure for all other $f_0$ $\in [0.2, 0.4, 0.5, 0.7, 0.9]$


### Exercise 3
$$
 x(n) = 1, 0 \lt n \lt 9\\
 y(n) = 2, 0 \lt n \lt 14
$$

Form the two sequences:

```matlab
start = 0, end_ = 30, interval = .2 % Change these!!

n = start: interval: end_t % Remember f = 301 points, length of n must equal length(f)
                    % Whatever your start, end_ is, remember that
                    % So if any changes, change the interval too
                    
                    % But you won't be skinned for having exact values as everyone
                    % else. Isikustresss!

x = zeros(n)
y = x

x(1:9) = 1
y(0:14) = 2

```

$$z(n) =  x(n) * y(n)$$
We'll do linear convolution

```matlab
z = conv(x, y)

```

DTFT of $z(n)$
```matlab
    f = linspace(-1.5, 1.5, 301)
    w = 2 * pi * f;

    X = dtft(n, x, w)
    Y = dtft(n, y, w)

    % Index for z(n) will be 1/2 the x(n), y(n) one
    n  = n(0): interval/2: n(length(n))
    
    Z = dtft(n, z, w) % A

    X_times_Y = X.* Y %B
    
    % Compare absolute values of A and B

    plot(abs(X_times_Y))
    plot(abs(Z))
```

# Lab 3: DFT and FFT
## Exercise 1

*Tukachokea hapo*