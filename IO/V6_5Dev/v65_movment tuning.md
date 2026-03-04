# Overview of the plan
The new system has several of the parameters working better with the light, but there are a few more details that I would like to implement around the movement of the light itself.

## Updateing current method
- Keep the wander box as the main bounds, but make the movement of the light within it much more related to the movement data instead of random points.
- Also, now that the oscillation of the falloff is working, it opens up an opportunity for the light to not necessarily move all the time. The oscilating falloff can still animate the lights.
