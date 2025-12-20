---
sidebar_position: 3
---

# Custom Models

Learn how to create your own dinosaur models for Mesozoic Labs.

## MuJoCo XML Format

Dinosaur models are defined using MuJoCo's XML format:

```xml
<mujoco model="custom_dino">
  <worldbody>
    <body name="torso" pos="0 0 1.5">
      <joint type="free"/>
      <geom type="capsule" size="0.3 0.5"/>

      <!-- Add limbs, tail, etc. -->
    </body>
  </worldbody>

  <actuator>
    <!-- Define motors for each joint -->
  </actuator>
</mujoco>
```

## Model Requirements

1. Bipedal or quadrupedal stance
2. Properly configured joints with limits
3. Actuators for each controllable joint
4. Appropriate mass distribution

:::note Coming Soon
Custom model creation guide is under development.
:::
