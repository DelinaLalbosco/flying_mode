import re
import matplotlib.pyplot as plt

XML_FILE = (
    "src/S10_sdk_deploy/"
    "S10_description/s10_mjcf/mjcf/S10_track.xml"
)

waypoints = []

with open(XML_FILE, "r") as f:
    for line in f:
        match = re.search(
            r'name="track_waypoint_(\d+)".*?pos="([^"]+)"',
            line
        )

        if match:
            index = int(match.group(1))
            pos = [float(x) for x in match.group(2).split()]

            waypoints.append(
                (index, pos[0], pos[1], pos[2])
            )

# Sort by waypoint number
waypoints.sort(key=lambda x: x[0])

if not waypoints:
    raise RuntimeError("No track_waypoint_* found in XML")

x = [p[1] for p in waypoints]
y = [p[2] for p in waypoints]

plt.figure(figsize=(10, 8))

# Track line
plt.plot(
    x,
    y,
    "-o",
    linewidth=1.5,
    markersize=5,
)

# Number every waypoint
for index, xi, yi, zi in waypoints:
    plt.annotate(
        str(index),
        (xi, yi),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("S10 Multi-Terrain Waypoint Track")

plt.axis("equal")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "s10_waypoint_track.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("\nWaypoint list:")
print("-" * 50)

for index, xi, yi, zi in waypoints:
    print(
        f"{index:2d}: "
        f"X={xi:8.4f}  "
        f"Y={yi:8.4f}  "
        f"Z={zi:8.4f}"
    )

print("\nSaved: s10_waypoint_track.png")