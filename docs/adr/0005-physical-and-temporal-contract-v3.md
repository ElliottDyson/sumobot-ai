# ADR 0005: physical and temporal arena contract v3

Status: accepted

Arena contract v2 accidentally applied the `40 x 40 x 80 mm` standardized dimensions to the chassis collision box.
Its 52 mm wheel track and 12 mm full wheel widths therefore produced a 64 mm external width, while chassis clearance
produced an 86 mm external height. Contract v3 treats `40 x 40 x 80 mm` as the complete legal robot envelope. The
chassis is `40 x 28 x 74 mm`, centred 43 mm above the board; a 34 mm track and 6 mm full wheel widths fit exactly
inside 40 mm. The rear support is a finite `8 x 8 x 2 mm` skid pad rather than a spherical point proxy.

The drivetrain remains a per-wheel velocity interface, but command latency, deadband, quantization, slew rate and
first-order response occur before the actuator target. Battery voltage, left/right mismatch, motor strength and a
torque-speed envelope affect physics. MuJoCo-Warp rolling and torsional friction are randomized for wheels and skid,
along with contact compliance, COM and inertia. Contact forces are collected from every physics substep and averaged
over the 50 Hz control interval; teacher contact fields are no longer zero placeholders.

Ring-out is based on complete loss of projected support. The two wheel contact points and skid point are transformed
with the chassis pose. If all three remain beyond the tabletop for 0.1 seconds, that robot is out. The existing
zero-sum winner/draw resolution and inactivity precedence remain unchanged.

The privileged teacher is still feedforward at the actor level, but its observation now contains a ten-step compact
history of its own proposed/executed commands, wheel and chassis motion, contact, edge margin and opponent-relative
motion. Opponent private commands are excluded. The deployable student has its own ten-step sensor/action history in
addition to CAP-Dreamer's recurrent RSSM, matching the released Extreme Parkour design in which finite proprioceptive
history and recurrent visual state coexist.

The deployable sensor generator produces quantized/delayed/noisy encoders, body-frame IMU signals, four finite-board
downward rays, and an FOV/range-limited opponent observation. It holds low-rate samples until the next delivery and
reports sensor age. All replay/world-model transitions continue to use the normalized executed command, never a raw
proposal or a teacher label.

Arena-v2 bootstrap checkpoints are intentionally incompatible because geometry, dynamics and the teacher observation
dimension changed. New runs use checkpoint version 2 and record arena/history dimensions in the checkpoint so an
incompatible resume fails explicitly.
