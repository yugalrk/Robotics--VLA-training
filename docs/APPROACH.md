# Kinematics-based data augmentation

This highlighted our main challenge: limited diversity in the training data, which motivated us to explore ways to generate more varied demonstrations.

Our Approach: Kinematics-Based Data Augmentation

Our solution reuses recorded robot movements and combines them with the robot's kinematic model to calculate new trajectories for different box positions, without training an additional AI model.

1. Data Exploration

We analyze the existing demonstrations, including camera recordings, joint states, and actions. Each robot state contains 16 values: seven joint angles and one gripper state per arm.

These recordings provide the original movement patterns used as references for trajectory generation.

2. Robot Kinematics and Trajectory Reconstruction

We combine the recorded joint angles with the FR3 Duo's kinematic model, which describes its link geometry, joint axes, and movement limits.

Two main algorithms are used:

Forward Kinematics (FK): Reconstructs the robot's original gripper positions and trajectories in 3D space from recorded joint angles.

Inverse Kinematics (IK): Calculates joint configurations needed to reach new target positions.

Our IK implementation uses Damped Least Squares (DLS) with joint-limit constraints and null-space optimization to keep calculated joint configurations close to the original recorded movements.

This allows us to estimate the robot's reachable workspace and adapt movements to new box locations while preserving the reference trajectory as much as possible.

3. Trajectory Retargeting and Lookup Table

We divide the tabletop into a 5 cm grid and calculate candidate left-arm trajectories for different box positions.

The process involves:

Mapping the selected box position from camera coordinates to the tabletop grid.

Adapting the original gripper trajectory to the new target.

Using inverse kinematics to calculate the corresponding joint movements.

Precomputing and storing the resulting trajectories in a Lookup Table (LUT) for fast retrieval.

Current results:

136 feasible grid positions with candidate left-arm trajectories.

26 grid positions marked as outside the reachable workspace under the implemented constraints.

Interactive visualization of the calculated trajectories for different box positions.

4. Toward Automated Training Data Generation

The next step is to execute and validate the adapted trajectories in Isaac Sim, including grasp success, collision checks, and the complete dual-arm handover.

Successful executions could then be recorded with synchronized camera observations, robot states, and actions to create new training demonstrations covering more diverse box positions.

These augmented demonstrations could be used to fine-tune imitation-learning models and evaluate whether increased data diversity improves generalization.
