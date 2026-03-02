# Overview
This interactive lighting installation is composed of 12 light panels in total.
4 units with 3 panels each. The brightness of each panel is determined by calculating a virtual brightness it recieves from a simulated point light within the scene. This is calculated by creating a virtual sensor point in the middel of each panel to read the light value from the simulated light. All changes to the system are done by changing the position and brightness properties of the singular simulated point light.
## Location
The installation is located inside a storefront at a busy intersection. The installation should be designed to respond to the following zones:
1. trackzone: the area directly outside the storefront. This area requires an active decision to engage with the installation. the outer edge moving away from the window is defined by 2 large pillars. the zone is centered on the middle of the 4 panels.
475 cm wide in X
205 cm deep in Z 
300 cm high in y

It begins 78 cms from the front of the panel in Z
It is 66 cm lower than the base of the panels in Y because the street is lower than the storefront
2. passivezone: the area beyond the pillars that define the trackzone changes as you move away from the camera input. 
- First is a busy sidewalk that will have many people walking past the exhibit, but not engaging with it directly. 
- Next is a buffer zone and then a bike lane.
- after the bike lane is 4 lanes of traffic on a busy road.


## Panel Physical Configuration

### Overview
The installation consists of 4 identical units, each containing 3 LED panels arranged in a Y-formation when viewed from the side.

### Panel Dimensions
- **Panel size**: 60cm × 60cm
- **Gap between units**: 20cm
- **Unit spacing**: 80cm center-to-center between units
- **Total installation width**: ~300cm (from center of Unit 0 to center of Unit 3 = 240cm, plus panel edges)

### Unit Structure (Side View)
```
         │ Panel 1 (vertical)
         │
         ▼
    ┌────┴────┐  Junction point
    │  3.5cm  │  (3.5cm vertical, 5cm horizontal offset)
   ╱    5cm    ╲
  ╱             ╲
 ╱   22.5°  22.5° ╲
╱                   ╲
Panel 2            Panel 3
(front/street)     (back, faces forward)
tilted UP          tilted DOWN
```

### Panel Positions Per Unit

| Panel | Description | Position (relative) | Angle from Vertical | Normal Direction |
|-------|-------------|---------------------|---------------------|------------------|
| Panel 1 | Top, vertical | y = 90cm, z = 0cm | 0° | Faces forward (+Z) |
| Panel 2 | Lower front | y = 30cm, z = +12cm | 22.5° forward | Outward and UP |
| Panel 3 | Lower back | y = 30cm, z = -12cm | 22.5° backward | Forward and DOWN |

### Unit X Positions
| Unit | X Position |
|------|------------|
| Unit 0 | 0cm |
| Unit 1 | 80cm |
| Unit 2 | 160cm |
| Unit 3 | 240cm |

### Virtual Sensor Positions (Panel Centers)
Each panel has a virtual sensor at its center for calculating brightness from the point light.

| Unit | Panel | Sensor Position (x, y, z) in centimeters |
|------|-------|------------------------------------------|
| 0 | 1 (top) | (0, 90, 0) |
| 0 | 2 (front) | (0, 30, 12) |
| 0 | 3 (back) | (0, 30, -12) |
| 1 | 1 (top) | (80, 90, 0) |
| 1 | 2 (front) | (80, 30, 12) |
| 1 | 3 (back) | (80, 30, -12) |
| 2 | 1 (top) | (160, 90, 0) |
| 2 | 2 (front) | (160, 30, 12) |
| 2 | 3 (back) | (160, 30, -12) |
| 3 | 1 (top) | (240, 90, 0) |
| 3 | 2 (front) | (240, 30, 12) |
| 3 | 3 (back) | (240, 30, -12) |

### Panel Normals (for lighting calculations)
Panel normals are unit vectors indicating the direction each panel faces:

| Panel | Normal Vector (x, y, z) | Notes |
|-------|-------------------------|-------|
| Panel 1 | (0, 0, 1) | Faces directly toward street |
| Panel 2 | (0, 0.38, 0.92) | Tilted UP, faces outward toward street |
| Panel 3 | (0, -0.38, 0.92) | Tilted DOWN, faces forward toward street |

### Coordinate System
- **X-axis**: Left to right along the installation (0 = leftmost unit)
- **Y-axis**: Vertical height (0 = base of panels, -66cm = street level)
- **Z-axis**: Depth, positive toward street, negative toward interior
- **Origin**: Base of panels at Unit 0
- **Units**: Centimeters

### DMX Mapping
Each unit controls 3 DMX channels (one per panel):
- Channels 1-3: Unit 0 (Panels 1, 2, 3)
- Channels 4-6: Unit 1 (Panels 1, 2, 3)
- Channels 7-9: Unit 2 (Panels 1, 2, 3)
- Channels 10-12: Unit 3 (Panels 1, 2, 3)

- to double check the dmx channel numbers consult artnetTest.py



## Point Light Controller
The virtual point light will have the following properties
### Properties

#### Position
Position shown in centimeters based on the 0,0,0 point of the scene. The goal is to establish a calibrated relationship between the virtual lights and physical panels 
- positionX
- - Units: centimeters
- positionY
- - Units: centimeters
- positionZ
- - Units: centimeters

#### Brightness
The brightness properties of the light are calcuated by the pulsing of the light. The light never stops pulsing.
- brightnessMin
- - Units: DMX .Lowest possible value:1 . Highest possible value: 50 
- brightnessMax
- - Units: DMX .Lowest possible value:1 . Highest possible value: 50
- pulseSpeed
- - Units: milliseconds. This should be the time that it takes to complete a full pulse cycle
- falloffRadius
- - Units: centimeters . The goal is to make the relationship spatial so this should be where it hits 0. falloff should be calculated linearly

#### Property behavior
- All changes should be animated. New input values are set as a target to be moved to. This includes new positions or changes to brightness.

### Narrative of Point Light behavior
The light should be treated like a simple animated character that is changing its properties based on inputs from the camera.(simulated for now). It should take passive inputs from the passive passersby on the sidewalk, from bicycles, and from the cars

#### Basic behavior   
When no-one is in the tracking zone, the virtual light should wander around the 3d area around the lights inside the building. This area should be defined by a clear 3d bounding box. The areas that it moves around , the speed, pulse speed, min/max brightness will be determined by the inputs of passive movement and the time of day.
When a person moves into the tracking zone the light should move to the center of their bounding box. when going into this mode, dynamic adjustments to the falloff radius will be needed so that the virtual light can still effect the panels

Overall the point light should be seen as a curious creature taking in the activity on the street.

#### Time of Day
The area is in a busy area in the financial district of Toronto so the time of day effects the corner very much. 
- Daytime should be much more active with the light moving freely around the interior volume
- Nightime should make it more likely to stay lower to the ground, usually only effecting panels 2,3 but with some ventures up or changes in falloff radius to respond to inputs.


