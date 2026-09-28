# Hardware acquisition

The capstone system used an Arduino Mega sensor node, Digi XBee/ZigBee transport, a Raspberry Pi base station, and ECG, temperature and SpO2 sensing. The report and poster describe that integrated system. Only the earlier single-channel AD8232 example was recovered as firmware in the reviewed folders.

## Recovered AD8232 sketch

Open `ad8232_demo/ad8232_demo.ino` in the Arduino IDE. The sketch uses A0 for the analog signal, pins 10/11 for lead-off detection, and serial at 9600 baud. It emits an ADC integer or `!` when a lead is disconnected. Original SparkFun attribution is preserved in the file.

```sh
python -m pip install -e '.[hardware]'
nrsaa capture --port /dev/ttyACM0 --output outputs/capture.csv --max-lines 15000
```

Replace the port with the board's actual serial port. Capture is bounded by a line limit and stops on a five-second read timeout. The output retains lead-off and invalid-message events, rather than silently treating them as signal values. Host monotonic timestamps record arrival times.

The sketch's `delay(1)` is not a calibrated sampling clock. Serial throughput at 9600 baud also limits the effective rate. Do not treat this stream as a 500 Hz recording or pass it directly to the three-lead classifier. The capture output is raw ADC counts, not millivolts; the board's actual calibration would be required for conversion.

Tests cover serial parsing with simulated input. Firmware compilation, board operation, electrical safety, and wireless integration were not validated during this cleanup. Hardware experiments should use a signal simulator and an appropriate isolated test setup.
