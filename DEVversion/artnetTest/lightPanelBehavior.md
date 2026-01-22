## Overview of Light Panel Behavior Documentation
This installation is designed as an installation inside a storefront at a busy intersection. The installation should act as an autonomous entity that is constantly reading information from the environment and adjusting its patterns. The installation runs 24 hours a day and should respond differently based on the time of day.

The installation is curious and responds to both the passive and active engagement of passersby. 

## Top Level Behavior Rules
1. No sudden changes in brightness. While there should be an noticeable change in speed of change in the patterns , the maximum speed of change should still be smooth. The max change speed should be a parameter that can be adjusted.
2. DMX stays between 1 and 50.
3. Panels 1 and 2 from each unit should work together as a single sub-unit. The back row of panel 3s, should be separate. However the to sub-units should communicate.
4. biggest influence should be from people moving in trackzone1 and trackzone2 to actively engage with the installation.
5. secondary influence should be from people moving in trackzone3 - trackzone10. These zones are passive zones that represent a busy sidewalk, a bike land, and a busy street. The installation should respond to the density of people in these zones but not be overly reactive.
6. Time of day and especially rush hours should have a strong influence on the overall brightness and speed of change of the installation.
- Specific attention should be given after 6pm when the area becomes much less busy from people

## Location
The installation is located inside a storefront at a busy intersection. The installation should be designed to respond to the following zones:
1. trackzone1: the area directly outside the storefront. This area requires an active decision to engage with the installation.
2. trackzone2: located in between the 2 large pillars outside the storefront. This is also an area that shows active engagement.
3. trackzone3 - trackzone10. are all passive zones. They represent a busy sidewalk, a bike land, and a busy street. 

## General Response Strategy
The installation should have a general library of responses :
1. Ambient - taking in passive inputs or fewer inputs
2. Acknowledge - noticing that someone is in an active zone
3. Engage - actively responding to people in active zones
4. Peak - responding to high density of people in active zones

However, there should be a smooth transition between these states. The installation should never feel like it is switching between modes but rather that it is constantly evolving based on the inputs it is receiving.








