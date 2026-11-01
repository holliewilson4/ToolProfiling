import io
import zipfile
from pathlib import Path

import cv2
import ezdxf
import numpy as np
import streamlit as st
from PIL import Image


# ------------------------------------------------------------
# Streamlit page and application settings
# ------------------------------------------------------------

st.set_page_config(page_title="Tool Profiling", page_icon="🛠️", layout="wide")

st.title("🛠️ Tool Profiling")
st.caption(
    "Upload a shadowbox image to generate individual tool DXFs. "
)

MIN_CONTOUR_AREA = 2500
DISPLAY_MAX_WIDTH = 1000
DISPLAY_MAX_HEIGHT = 650
UPLOAD_PREVIEW_WIDTH = 560
UPLOAD_PREVIEW_HEIGHT = 360
LIGHTBED_PREVIEW_WIDTH = 760
LIGHTBED_PREVIEW_HEIGHT = 560

# The corrected image is generated at this resolution.
# DXF dimensions are still exported in millimetres.
OUTPUT_PIXELS_PER_MM = 3.0


# ------------------------------------------------------------
# Image loading and display
# ------------------------------------------------------------

def read_image(uploaded_file):
    data = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("The uploaded file could not be read as an image.")

    return image


def display_sized(image, max_width=DISPLAY_MAX_WIDTH, max_height=DISPLAY_MAX_HEIGHT):
    h, w = image.shape[:2]
    factor = min(max_width / w, max_height / h, 1.0)

    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ------------------------------------------------------------
# Automatic lightbed detection
# ------------------------------------------------------------

def order_corners(points):
    """
    Return four corners in this order:
    top-left, top-right, bottom-right, bottom-left.
    """
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(sums)]          # top-left
    ordered[2] = points[np.argmax(sums)]          # bottom-right
    ordered[1] = points[np.argmin(differences)]   # top-right
    ordered[3] = points[np.argmax(differences)]   # bottom-left

    return ordered


def quadrilateral_dimensions(corners):
    tl, tr, br, bl = corners

    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)

    average_width = (top + bottom) / 2
    average_height = (left + right) / 2

    return average_width, average_height


def find_bed_corners(image, bed_width_mm, bed_height_mm):
    """
    Try to locate the rectangular lightbed automatically.

    The detector searches image edges and several thresholded versions
    of the image for a large, convex four-corner contour. It ranks
    candidates using their area and approximate aspect ratio.

    This works best when the lightbed has a visible border that contrasts
    with the surrounding surface.
    """
    image_height, image_width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    target_ratio = bed_width_mm / bed_height_mm
    candidates = []

    # Several detection methods help with different lighting conditions.
    edge_versions = []

    edges = cv2.Canny(blurred, 40, 130)
    edge_versions.append(edges)

    for threshold_value in (70, 100, 130, 160, 190, 220):
        _, threshold = cv2.threshold(
            blurred,
            threshold_value,
            255,
            cv2.THRESH_BINARY
        )
        edge_versions.append(threshold)

        _, threshold_inverse = cv2.threshold(
            blurred,
            threshold_value,
            255,
            cv2.THRESH_BINARY_INV
        )
        edge_versions.append(threshold_inverse)

    kernel = np.ones((5, 5), np.uint8)

    for version in edge_versions:
        closed = cv2.morphologyEx(
            version,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            image_area = image_width * image_height
            area_ratio = area / image_area

            # Ignore tiny objects and contours that are effectively
            # the entire photograph.
            if area_ratio < 0.12 or area_ratio > 0.97:
                continue

            perimeter = cv2.arcLength(contour, True)

            for epsilon_factor in (0.015, 0.02, 0.03, 0.04):
                approx = cv2.approxPolyDP(
                    contour,
                    epsilon_factor * perimeter,
                    True
                )

                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue

                corners = order_corners(approx.reshape(4, 2))
                candidate_width, candidate_height = quadrilateral_dimensions(corners)

                if candidate_width <= 0 or candidate_height <= 0:
                    continue

                candidate_ratio = candidate_width / candidate_height

                # Compare the candidate's apparent ratio with the
                # dimensions entered by the user.
                ratio_error = abs(np.log(candidate_ratio / target_ratio))

                # Prefer large candidates with a plausible aspect ratio.
                score = area_ratio * np.exp(-2.0 * ratio_error)

                candidates.append((score, corners))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_corners = candidates[0]

    # A very poor match is more likely to be a false detection.
    if best_score < 0.08:
        return None

    return best_corners


def crop_and_calibrate(image, corners, bed_width_mm, bed_height_mm):
    """
    Perspective-correct and crop the photograph to the detected bed.

    The output image is generated at a fixed pixels-per-mm resolution.
    This makes the pixel-to-mm conversion consistent in both directions.
    """
    output_width = max(1, int(round(bed_width_mm * OUTPUT_PIXELS_PER_MM)))
    output_height = max(1, int(round(bed_height_mm * OUTPUT_PIXELS_PER_MM)))

    source = np.asarray(corners, dtype=np.float32)

    destination = np.float32([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ])

    transform = cv2.getPerspectiveTransform(source, destination)

    corrected = cv2.warpPerspective(
        image,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC
    )

    # Since output pixels are generated at OUTPUT_PIXELS_PER_MM,
    # this is the exact scale used for DXF export.
    scale_mm_per_pixel = 1.0 / OUTPUT_PIXELS_PER_MM

    return corrected, scale_mm_per_pixel


def draw_bed_detection(image, corners, bed_width_mm, bed_height_mm):
    """
    Create a top-down, greyscale lightbed preview with 10 mm of surrounding
    photograph visible on every side. The green rectangle marks the detected
    bed boundary; the surrounding area lets the user check what was excluded.
    """
    margin_mm = 10.0
    preview_pixels_per_mm = 1.5

    margin_px = int(round(margin_mm * preview_pixels_per_mm))
    bed_width_px = max(1, int(round(bed_width_mm * preview_pixels_per_mm)))
    bed_height_px = max(1, int(round(bed_height_mm * preview_pixels_per_mm)))
    output_width = bed_width_px + 2 * margin_px
    output_height = bed_height_px + 2 * margin_px

    source = np.asarray(corners, dtype=np.float32)
    destination = np.float32([
        [margin_px, margin_px],
        [margin_px + bed_width_px - 1, margin_px],
        [margin_px + bed_width_px - 1, margin_px + bed_height_px - 1],
        [margin_px, margin_px + bed_height_px - 1]
    ])

    transform = cv2.getPerspectiveTransform(source, destination)
    context = cv2.warpPerspective(
        image,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(35, 35, 35)
    )

    gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Mark the actual detected lightbed edge, leaving the 10 mm context visible.
    cv2.rectangle(
        preview,
        (margin_px, margin_px),
        (margin_px + bed_width_px - 1, margin_px + bed_height_px - 1),
        (0, 200, 0),
        2
    )
    return preview


# ------------------------------------------------------------
# Tool contour processing
# ------------------------------------------------------------

def create_mask(image, threshold_value=140):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, threshold = cv2.threshold(
        blur,
        int(threshold_value),
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel
    )
    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    return threshold


def extract_tools(mask, min_area):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    valid = [
        contour for contour in contours
        if cv2.contourArea(contour) >= min_area
    ]

    return sorted(
        valid,
        key=lambda contour: cv2.boundingRect(contour)[0]
    )


def simplify_contour(contour, factor):
    epsilon = factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


# ------------------------------------------------------------
# DXF and ZIP export
# ------------------------------------------------------------

def contour_points(contour, scale):
    points = [
        (
            float(point[0][0] * scale),
            float(-point[0][1] * scale)
        )
        for point in contour
    ]

    if points:
        points.append(points[0])

    return points


def make_dxf(contours, names, scale):
    doc = ezdxf.new("R2010")
    modelspace = doc.modelspace()

    for contour in contours:
        points = contour_points(contour, scale)

        if len(points) >= 4:
            modelspace.add_lwpolyline(points, close=True)

    buffer = io.StringIO()
    doc.write(buffer)

    return buffer.getvalue().encode("utf-8")


def make_single_dxf(contour, scale):
    return make_dxf([contour], [""], scale)


def make_zip(contours, names, scale):
    memory = io.BytesIO()

    with zipfile.ZipFile(
        memory,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_file:

        for contour, name in zip(contours, names):
            safe_name = "".join(
                character if character.isalnum() or character in "-_ "
                else "_"
                for character in name
            ).strip()

            safe_name = safe_name or "Tool"

            zip_file.writestr(
                f"{safe_name}_dxf.dxf",
                make_single_dxf(contour, scale)
            )

    memory.seek(0)
    return memory.getvalue()


def annotate_image(corrected, contours, names):
    # Keep the final preview monochrome so tool edges stand out clearly.
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for contour, name in zip(contours, names):
        cv2.drawContours(
            output,
            [contour],
            -1,
            (0, 0, 255),
            3
        )

        x, y, width, height = cv2.boundingRect(contour)

        cv2.putText(
            output,
            name,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    return output


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

if "processed" not in st.session_state:
    st.session_state.processed = None

if "upload_id" not in st.session_state:
    st.session_state.upload_id = None


# ------------------------------------------------------------
# User interface
# ------------------------------------------------------------



st.subheader("1. Upload your shadowbox image")

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded:
    try:
        original = read_image(uploaded)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    current_id = f"{uploaded.name}:{uploaded.size}"

    if st.session_state.upload_id != current_id:
        st.session_state.upload_id = current_id
        st.session_state.processed = None
        st.session_state.pop("tool_names", None)
        st.session_state.pop("names_for", None)
        st.session_state.pop("rejected", None)

    st.write("Selected Image")
    st.image(
        display_sized(
            original,
            max_width=UPLOAD_PREVIEW_WIDTH,
            max_height=UPLOAD_PREVIEW_HEIGHT
        ),
        width=UPLOAD_PREVIEW_WIDTH
    )

    st.subheader("2. Lightbed Dimensions")

    dimension_col1, dimension_col2 = st.columns(2)

    with dimension_col1:
        bed_width_mm = st.number_input(
            "Width (mm)",
            min_value=50.0,
            max_value=3000.0,
            value=440.0,
            step=10.0
        )

    with dimension_col2:
        bed_height_mm = st.number_input(
            "Height (mm)",
            min_value=50.0,
            max_value=3000.0,
            value=320.0,
            step=10.0
        )

    st.caption(
        "Enter the width and height of the lightbed. "
    )
    st.subheader("3. Detect, Crop and Calibrate the lightbed")

    st.write(
        " "
    )

    if st.button("Detect lightbed and calibrate", type="primary"):
        with st.spinner("Locating the lightbed and correcting the photograph..."):
            try:
                corners = find_bed_corners(
                    original,
                    bed_width_mm,
                    bed_height_mm
                )

                if corners is None:
                    st.session_state.processed = None
                    st.error(
                        "The lightbed could not be identified confidently. "
                        "Try a photograph with a clearer view of all four bed edges, "
                        "less glare, and less background around the bed."
                    )
                else:
                    corrected, scale = crop_and_calibrate(
                        original,
                        corners,
                        bed_width_mm,
                        bed_height_mm
                    )

                    st.session_state.processed = {
                        "corrected": corrected,
                        "corners": corners,
                        "scale": scale,
                        "bed_width_mm": bed_width_mm,
                        "bed_height_mm": bed_height_mm,
                        "image_name": Path(uploaded.name).stem,
                        "upload_id": current_id
                    }

                    st.rerun()

            except Exception as exc:
                st.session_state.processed = None
                st.error(f"Automatic calibration failed: {exc}")

    result = st.session_state.processed

    if result and result.get("upload_id") == current_id:
        corrected = result["corrected"]
        scale = result["scale"]
        image_name = result["image_name"]

        st.success("Lightbed detected and image automatically calibrated.")

        st.write("Lightbed detection")
        detection_preview = draw_bed_detection(
            original,
            result["corners"],
            result["bed_width_mm"],
            result["bed_height_mm"]
        )
        st.image(
            display_sized(
                detection_preview,
                max_width=LIGHTBED_PREVIEW_WIDTH,
                max_height=LIGHTBED_PREVIEW_HEIGHT
            ),
            width=LIGHTBED_PREVIEW_WIDTH
        )

        st.info(
            f"Calibrated dimensions: {result['bed_width_mm']:.1f} mm × "
            f"{result['bed_height_mm']:.1f} mm. "
            f"Scale: {scale:.4f} mm/pixel."
        )

        st.subheader("4. Detect tools")

        with st.expander("Adjust processing settings"):
            min_area = st.number_input(
                "Minimum contour area (pixels²)",
                min_value=100,
                max_value=100000,
                value=MIN_CONTOUR_AREA,
                step=500
            )

            simplify_factor = st.number_input(
                "Outline simplification (higher = fewer points)",
                min_value=0.0001,
                max_value=0.02,
                value=0.0008,
                step=0.0001,
                format="%.4f"
            )

            threshold_value = st.slider(
                "Image threshold",
                min_value=0,
                max_value=255,
                value=140
            )

        if st.button("Detect tools", type="primary"):
            try:
                mask = create_mask(
                    corrected,
                    threshold_value
                )

                raw_contours = extract_tools(
                    mask,
                    int(min_area)
                )

                contours = [
                    simplify_contour(contour, float(simplify_factor))
                    for contour in raw_contours
                ]

                st.session_state.pop("rejected", None)

                st.session_state.processed.update({
                    "contours": contours,
                    "min_area": int(min_area),
                    "simplify_factor": float(simplify_factor),
                    "threshold_value": int(threshold_value),
                    "tools_detected": True
                })

                st.rerun()

            except Exception as exc:
                st.error(f"Could not detect tools: {exc}")

        result = st.session_state.processed

        if result.get("tools_detected"):
            contours = result["contours"]

            st.subheader("5. Review and Name the Detected Tools")

            if not contours:
                st.warning(
                    "No tools were detected. Try lowering the minimum contour "
                    "area or adjusting the image threshold."
                )

            else:
                st.success(
                    f"Detected {len(contours)} tool(s). "
                    "Check that the red generated outlines follow the "
                    "physical outline of the tools."
                )

                if (
                    "tool_names" not in st.session_state
                    or st.session_state.get("names_for") != current_id
                    or len(st.session_state.tool_names) != len(contours)
                ):
                    st.session_state.tool_names = [
                        f"Tool_{index + 1}"
                        for index in range(len(contours))
                    ]
                    st.session_state.names_for = current_id

                # Indices of tools the user has rejected with the X button.
                rejected = st.session_state.setdefault("rejected", set())
                kept = [i for i in range(len(contours)) if i not in rejected]

                # Display the labelled image before asking the user to name
                # each detected tool.
                preview = annotate_image(
                    corrected,
                    [contours[i] for i in kept],
                    [f"Tool_{i + 1}" for i in kept]
                )

                st.subheader("Labelled image")
                st.image(
                    display_sized(
                        preview,
                        max_width=DISPLAY_MAX_WIDTH,
                        max_height=DISPLAY_MAX_HEIGHT
                    ),
                    width=DISPLAY_MAX_WIDTH
                )

                st.subheader("Name your tools")
                naming_columns = st.columns(2)

                for index in kept:
                    with naming_columns[index % 2]:
                        name_col, x_col = st.columns(
                            [6, 1],
                            vertical_alignment="bottom"
                        )

                        with name_col:
                            st.session_state.tool_names[index] = st.text_input(
                                f"Tool {index + 1}",
                                value=st.session_state.tool_names[index],
                                key=f"name_{current_id}_{index}"
                            )

                        with x_col:
                            st.button(
                                "❌",
                                key=f"reject_{current_id}_{index}",
                                help="Remove this outline",
                                on_click=rejected.add,
                                args=(index,)
                            )

                names = [
                    st.session_state.tool_names[i].strip() or f"Tool_{i + 1}"
                    for i in kept
                ]
                kept_contours = [contours[i] for i in kept]

                # Refresh the labels using the names entered above. This
                # remains the image used for the final download.
                preview = annotate_image(
                    corrected,
                    kept_contours,
                    names
                )

                st.subheader("6. Export files")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.download_button(
                        "Download combined DXF",
                        data=make_dxf(kept_contours, names, scale),
                        file_name=f"{image_name}_entire_bed.dxf",
                        mime="application/dxf",
                        use_container_width=True
                    )

                with col2:
                    st.download_button(
                        "Download individual DXFs (ZIP)",
                        data=make_zip(kept_contours, names, scale),
                        file_name=f"{image_name}_individual_tools.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

                with col3:
                    success, encoded = cv2.imencode(".jpg", preview)

                    if success:
                        st.download_button(
                            "Download contoured image",
                            data=encoded.tobytes(),
                            file_name=f"{image_name}_Contoured.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )

                st.caption(
                    "Downloads are generated by the browser. "
                )


st.divider()
st.caption(
    "Inspect the detected boundary and DXF outlines before using "
    "the exported files for machining."
)
