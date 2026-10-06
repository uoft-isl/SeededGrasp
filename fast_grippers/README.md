# Fast Grippers

Copies of the gripper .usd files in [grippers](../grippers) used for the 6-axis (`axes_forces`) test in the [cluttered scene pipeline](../docs/scene_generation.md). Compared to the originals:

- `physxJoint:maxJointVelocity` is raised on every joint (1000 for Barrett, 600 for Allegro, 500 for all other grippers), so the fingers reach their targets almost immediately.
- franka_panda: finger drives use stiffness 1000, damping 100 and max force 100 (instead of 1e7, 1e5 and 25), joint projection is disabled and the finger masses are 0.01 kg (instead of 0.1 kg).
- [gripper_isaac_info.json](gripper_isaac_info.json): the Barrett hand runs at a physics frequency of 240 Hz instead of 120 Hz.

Only the .usd files needed by the simulation are included. URDFs, meshes and [controller_info.json](../grippers/controller_info.json) are read from [grippers](../grippers). Available grippers: Allegro, Barrett, HumanHand, franka_panda, jaco_robot, robotiq_3finger, sawyer, shadow_hand, wsg_50.
