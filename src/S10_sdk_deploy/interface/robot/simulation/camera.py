
import cv2
import mujoco
import numpy as np
from pathlib import Path

# ============================================================
# MuJoCo model
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

MJCF_DIR = (
    CURRENT_DIR
    / ".."
    / ".."
    / ".."
    / "S10_description"
    / "s10_mjcf"
    / "mjcf"
).resolve()

XML_PATH = MJCF_DIR / "S10_track.xml"

# ============================================================
# Camera
# ============================================================

CAMERA_NAME = "front_camera"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# ============================================================
# Green detection
# ============================================================

# We will initially use a VERY broad green range.
LOWER_GREEN = np.array([20, 30, 20], dtype=np.uint8)
UPPER_GREEN = np.array([100, 255, 255], dtype=np.uint8)

# ============================================================
# Main
# ============================================================

def main():

    print("==============================================")
    print("S10 CAMERA DEBUG")
    print("==============================================")

    print()
    print("Loading MuJoCo model:")
    print(XML_PATH)
    print()

    if not XML_PATH.exists():
        print("ERROR: S10_track.xml not found!")
        return

    # --------------------------------------------------------
    # Load MuJoCo model
    # --------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )

    data = mujoco.MjData(model)

    print("MuJoCo model loaded.")
    print(f"Model nq = {model.nq}")
    print(f"Model nv = {model.nv}")
    print(f"Model nu = {model.nu}")
    print()

    # --------------------------------------------------------
    # Set robot initial position
    # --------------------------------------------------------

    data.qpos[:3] = np.array([
        0.0,
        -2.5,
        0.2
    ])

    # Normal orientation
    # w x y z
    data.qpos[3:7] = np.array([
        1.0,
        0.0,
        0.0,
        0.0
    ])

    mujoco.mj_forward(
        model,
        data
    )

    print("Robot initial pose:")
    print("Position:", data.qpos[:3])
    print("Quaternion:", data.qpos[3:7])
    print()

    print("Robot is FIXED.")
    print("MuJoCo physics stepping is disabled.")
    print()

    # --------------------------------------------------------
    # List cameras
    # --------------------------------------------------------

    print("Available cameras:")

    for i in range(model.ncam):

        camera_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            i
        )

        print(
            f"ID {i}: {camera_name}"
        )

    print()

    # --------------------------------------------------------
    # Find front camera
    # --------------------------------------------------------

    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        CAMERA_NAME
    )

    if camera_id < 0:

        print(
            f"ERROR: Camera '{CAMERA_NAME}' not found!"
        )

        return

    print(
        f"Front camera found. ID = {camera_id}"
    )

    # --------------------------------------------------------
    # Camera information
    # --------------------------------------------------------

    camera_position = data.cam_xpos[camera_id].copy()

    camera_rotation = data.cam_xmat[
        camera_id
    ].reshape(3, 3)

    print()
    print("Camera position:")
    print(camera_position)

    print()
    print("Camera rotation matrix:")
    print(camera_rotation)

    # MuJoCo camera looks along local -Z.
    camera_forward = -camera_rotation[:, 2]

    camera_up = camera_rotation[:, 1]

    camera_right = camera_rotation[:, 0]

    print()
    print("Camera forward direction:")
    print(camera_forward)

    print()
    print("Camera up direction:")
    print(camera_up)

    print()
    print("Camera right direction:")
    print(camera_right)

    print()
    print("==============================================")

    # --------------------------------------------------------
    # Renderer
    # --------------------------------------------------------

    renderer = mujoco.Renderer(
        model,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH
    )

    print("Camera renderer created.")
    print()

    print("Starting camera...")
    print()
    print("Robot is NOT moving.")
    print("We are ONLY testing the camera.")
    print()
    print("You should see TWO windows:")
    print()
    print("1. S10 Front Camera")
    print("2. Green Mask")
    print()
    print("Controls:")
    print("Q = quit")
    print("S = save current image")
    print("==============================================")
    print()

    # ========================================================
    # Main camera loop
    # ========================================================

    frame_count = 0

    while True:

        # ----------------------------------------------------
        # IMPORTANT:
        # Do NOT step physics.
        #
        # The robot remains completely fixed.
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Update camera scene
        # ----------------------------------------------------

        renderer.update_scene(
            data,
            camera=camera_id
        )

        # ----------------------------------------------------
        # Render image
        # ----------------------------------------------------

        image = renderer.render()

        # MuJoCo gives RGB
        # OpenCV uses BGR

        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        # ====================================================
        # GREEN DETECTION
        # ====================================================

        hsv = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2HSV
        )

        green_mask = cv2.inRange(
            hsv,
            LOWER_GREEN,
            UPPER_GREEN
        )

        # ----------------------------------------------------
        # Remove very small noise
        # ----------------------------------------------------

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        green_mask = cv2.morphologyEx(
            green_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        green_mask = cv2.morphologyEx(
            green_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # ----------------------------------------------------
        # Find green pixels
        # ----------------------------------------------------

        ys, xs = np.where(
            green_mask > 0
        )

        green_pixel_count = xs.size

        # ====================================================
        # DETECTION RESULT
        # ====================================================

        if green_pixel_count > 50:

            green_center_x = int(
                np.mean(xs)
            )

            green_center_y = int(
                np.mean(ys)
            )

            image_center_x = IMAGE_WIDTH // 2
            image_center_y = IMAGE_HEIGHT // 2

            error_x = (
                green_center_x
                - image_center_x
            )

            error_y = (
                green_center_y
                - image_center_y
            )

            print(
                f"[CAMERA] "
                f"Green X={green_center_x}, "
                f"Green Y={green_center_y}, "
                f"Error X={error_x}, "
                f"Error Y={error_y}, "
                f"Pixels={green_pixel_count}"
            )

            # ------------------------------------------------
            # Draw detected green center
            # ------------------------------------------------

            cv2.circle(
                image_bgr,
                (
                    green_center_x,
                    green_center_y
                ),
                10,
                (0, 0, 255),
                -1
            )

            # ------------------------------------------------
            # Draw image center
            # ------------------------------------------------

            cv2.line(
                image_bgr,
                (
                    image_center_x,
                    0
                ),
                (
                    image_center_x,
                    IMAGE_HEIGHT
                ),
                (255, 0, 0),
                2
            )

            cv2.line(
                image_bgr,
                (
                    0,
                    image_center_y
                ),
                (
                    IMAGE_WIDTH,
                    image_center_y
                ),
                (255, 0, 0),
                2
            )

            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            cv2.putText(
                image_bgr,
                f"Green X: {green_center_x}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                image_bgr,
                f"Green Y: {green_center_y}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                image_bgr,
                f"Error X: {error_x}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                image_bgr,
                f"Green pixels: {green_pixel_count}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:

            print(
                f"[CAMERA] "
                f"Green not detected. "
                f"Green pixels={green_pixel_count}"
            )

            cv2.putText(
                image_bgr,
                "GREEN NOT DETECTED",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.putText(
                image_bgr,
                f"Green pixels: {green_pixel_count}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # ====================================================
        # Display raw camera
        # ====================================================

        cv2.imshow(
            "S10 Front Camera",
            image_bgr
        )

        # ====================================================
        # Display green mask
        # ====================================================

        cv2.imshow(
            "Green Mask",
            green_mask
        )

        # ====================================================
        # Keyboard
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # Q = quit
        # ----------------------------------------------------

        if key == ord("q"):

            break

        # ----------------------------------------------------
        # S = save image
        # ----------------------------------------------------

        if key == ord("s"):

            filename = (
                f"camera_test_{frame_count}.png"
            )

            cv2.imwrite(
                filename,
                image_bgr
            )

            print()
            print(
                f"[CAMERA] Image saved: {filename}"
            )
            print()

        frame_count += 1

    # ========================================================
    # Cleanup
    # ========================================================

    renderer.close()

    cv2.destroyAllWindows()

    print()
    print("Camera stopped.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()

