"""Shared code for the FR3 Duo project.

data           dataset paths, train/val split, LeRobot loader
kinematics     forward kinematics of both arms (scene chain = official FR3v2 geometry)
retarget       IK retargeting of the recorded left-arm motion to a moved box
head_camera    head-camera model: world <-> pixel, pixel -> table plane
policy_runner  load any LeRobot checkpoint and run it (+ TCP server main)
subtasks       per-frame sub-task segmentation + quality checks
"""
