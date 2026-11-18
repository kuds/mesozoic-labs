"""Recheck of PR #669's design-package arithmetic (review, 18 Nov 2026). Prints each figure the errata call checked."""
g = 9.81
front = 0.55 * 15 * g                              # front axle load, N
print(f"front axle {front:.1f} N; rear {0.45 * 15 * g:.1f} N")
el = front * 0.179                                  # elbow moment, 179 mm arm
print(f"elbow {el:.2f} N·m; x1.5 {1.5 * el:.1f}; at the motor /0.9 {1.5 * el / 0.9:.2f} N·m")
print(f"rear at the motor {0.45 * 15 * g * 0.179 * 1.5 / 0.9:.2f} N·m")
print(f"forefoot-lift moment about the hind feet {15 * g * 0.358:.1f} N·m")
print(f"run time {111 * 0.8 / 210 * 60:.1f}-{111 * 0.8 / 150 * 60:.1f} min")
print(f"mass budget sum {6.16 + 2.70 + 1.00 + 0.95 + 0.75 + 0.70 + 0.90 + 1.84:.2f} kg")
