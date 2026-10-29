"""Angle between the global S1 frame and the local image-right/image-up ground directions at
different anchor pixels of the LIBERO living-room agentview (no roll). Read-only math."""
import numpy as np
F, C0 = 618.0386719675123, 255.5
R = np.array([[-1.7233905013069872e-06, -0.5287697435529835, 0.8487653140297038],
              [0.9999999999985034, -7.823149652530503e-07, 1.5430955956352577e-06],
              [-1.5194045527300304e-07, 0.848765314031093, 0.5287697435535403]])
P = np.array([0.6065773716836134, 0.0, 0.96])
Z = 0.44
def ray(uv):
    d = R @ np.array([(uv[0] - C0) / F, -(uv[1] - C0) / F, -1.0]); return P + (Z - P[2]) / d[2] * d
def unit(v): return v / np.linalg.norm(v)
right = unit(R[:2, 0]); away = np.array([-right[1], right[0]])
for uv in [(261, 442), (380, 298), (161, 274), (40, 300), (480, 300), (480, 480), (40, 480), (256, 200)]:
    p = ray(uv); r = unit((ray((uv[0] + 3, uv[1])) - p)[:2]); u = unit((ray((uv[0], uv[1] - 3)) - p)[:2])
    print(uv, "ground", np.round(p[:2], 3), "right_err %.1f" % np.degrees(np.arccos(np.clip(r @ right, -1, 1))),
          "behind_err %.1f" % np.degrees(np.arccos(np.clip(u @ away, -1, 1))))
