from src.distortion.restore_signal import restore_from_serial, plot_restoration

# k заносится вручную после калибровки по зебре
k_calibrated = 0.1185

_, signal, x_new_px, restored_signal = restore_from_serial(
    port='/dev/ttyUSB0',
    baudrate=1000000,
    k=k_calibrated,
    output_step_px=1.0,
)

plot_restoration(signal, x_new_px, restored_signal)