def keep_camera_frame_temporarily(frame):
    if frame is None:
        raise ValueError("Brak obrazu z kamery.")

    return frame.copy()