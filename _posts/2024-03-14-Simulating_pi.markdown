---
layout: post
title: "Piece of Pi: Estimating Pi by Monte Carlo Simulation"
excerpt: "What better way to celebrate Pi Day than by exploring the fascinating world of pi through a hands-on project? In this blog post, we'll dive into a simple yet engaging Python application that utilizes Monte Carlo simulations to estimate the value of pi."
categories: [Programming]
tags: [mathematics, pi]
comments: true
image:
  feature: 
  credit: 
  creditlink: 
---

What better way to celebrate Pi Day than by exploring the fascinating world of pi through a hands-on project? In this blog post, we'll dive into a simple yet engaging Python application that utilizes Monte Carlo simulations to estimate the value of pi.

## What is a Monte Carlo simulation?
Monte Carlo simulation is a statistical technique used to approximate the outcome of complex systems or processes by repeatedly sampling random inputs. It's named after the famous casino city in Monaco, renowned for its games of chance. The method involves generating random numbers and analyzing their distribution to make predictions or estimations.


## Estimating Pi
We will start by taking a quarter circle that fits into a square. The area of the quarter circle is given by:

<img src="/img/posts/Simulating_pi/Surface_circle.png" width="8%" height="auto">

The surface area of the square that encompasses this circle is r<sup>2</sup>. Thus, dividing the area of the square by that of the circle will give you 1/4<sup>th</sup> pi. We can estimate the ratio of the areas by taking n random points and looking how many lie inside the circle. The ratio of the areas is then equal to the number of points inside the circle divided by the total number of points n. Multiply by 4 and we get our estimate of pi.


## The Python Application
To implement this, we've created a simple Python application with a graphical user interface (GUI) using Tkinter. The UI allows users to specify the number of points to simulate and visualize the process of randomly scattering points and determining their location relative to the inscribed circle. The entire code can be found [here](https://github.com/MatthN/pieceofpi).

There are two main components to this code:
1. A generator that creates random points, checks if the fall inside or outside the quarter circle and updates the count.
2. A GUI component that adds each new random point to the visualization.

This is the code for the first component.

```python
import random
import math
from typing import Generator, Tuple

def monte_carlo_pi_generator(
        n_samples: int
) -> Generator[Tuple[Tuple[float, float, int], int, int], None, None]:
    inside_count = 0
    for i in range(1, n_samples + 1):
        x, y = random.random(), random.random()
        d = math.sqrt(x**2 + y**2)
        inside = 1 if d <= 1 else 0
        inside_count += inside
        yield (x, y, inside), inside_count, i
```

The generator creates a point with coordinates (x,y). Via Pythagoras' theorem we can check if the distance of this point from the center of the circle is larger than the radius of our circle (which is 1). If so, it is outside. Otherwise inside. We update the count and yield the point info together with the inside count and total count.

In the class that creates the GUI there is a method that uses this info to add the point to the visualization and update the estimate of pi.

```python
def simulate_pi_partially(self,
                          generator: Generator[Tuple[Tuple[float, float, int],
                                                      int, int], None, None]):
if self.paused_:
    # If paused, wait for a bit and then check again
    self.drawing_canvas_.after(100, self.simulate_pi_partially, generator)
else:
    try:
        point_info, inside_count, total_points = next(generator)
        (x, y, inside) = point_info
        pi_estimate = 4 * inside_count / total_points
        smoothed_pi = self.smooth_pi_estimate(pi_estimate)
        color = 'red' if inside else 'green'
        self.drawing_canvas_.create_oval(x * self.canvas_size,
                                        y * self.canvas_size,
                                        x * self.canvas_size + self.point_size,
                                        y * self.canvas_size + self.point_size,
                                        fill=color, outline=color)
        self.result_label_.config(text=f"Pi Estimate: {smoothed_pi:.10f}")
        self.points_counter_label_.config(text=f"Points: {total_points}")
        self.drawing_canvas_.after(1, self.simulate_pi_partially,
                                generator)
    except StopIteration:
        pass
```

Here you can see that we calculate the estimate of pi as 4 times the inside count over the total count. We add the point to the graph and give it a color depending on whether it falls inside or outside the circle.

<img src="/img/posts/Simulating_pi/App_simulation.png" width="50%" height="auto">



## Convergence
Although this method of estimating pi is simple and visual, it is not very fast in its convergence to the actual value of pi. Getting the first 4 digits alone already requires many million points (you can get lucky of course). Maybe this project triggered you to explore other ways of estimating pi. If so, do share!







