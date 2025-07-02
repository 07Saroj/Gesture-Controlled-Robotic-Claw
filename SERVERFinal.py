import asyncio
import websockets
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# Servo Configuration
SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500
OPEN_ANGLE = 0       # Claw fully open
CLOSE_ANGLE = 105    # Claw fully closed
MIN_SAFE_ANGLE = 20  # Prevents servo from straining
STEP_DELAY = 0.01    # Smooth movement delay
MAX_CURRENT_ANGLE = CLOSE_ANGLE - 15  # Safety margin

# Initialize PCA9685
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# Servo setup with pulse width limits
claw_servo = servo.Servo(pca.channels[0], min_pulse=SERVO_MIN_PULSE, max_pulse=SERVO_MAX_PULSE)
y_servo = servo.Servo(pca.channels[1], min_pulse=SERVO_MIN_PULSE, max_pulse=SERVO_MAX_PULSE)
x_servo = servo.Servo(pca.channels[3], min_pulse=SERVO_MIN_PULSE, max_pulse=SERVO_MAX_PULSE)

# Global state
current_claw_angle = OPEN_ANGLE
claw_active = False

async def smooth_move(servo, target_angle, min_angle=0, max_angle=180):
    """Safely moves servo with angle limits"""
    global current_claw_angle
    
    if servo == claw_servo:
        target_angle = max(min_angle, min(target_angle, max_angle))
        if not claw_active:
            return  # Don't move if claw is inactive
    
    current = servo.angle if servo.angle is not None else min_angle
    step = 1 if target_angle > current else -1
    
    while abs(current - target_angle) > 1:
        current += step
        servo.angle = current
        if servo == claw_servo:
            current_claw_angle = current
        await asyncio.sleep(STEP_DELAY)
    
    servo.angle = target_angle
    if servo == claw_servo:
        current_claw_angle = target_angle

async def handle_client(websocket):
    global claw_active
    
    async for message in websocket:
        try:
            parts = message.split(',')
            if len(parts) == 4:
                openness, x_angle, y_angle, active = map(float, parts)
                claw_active = bool(active)
                
                # Convert openness to angle with safety limits
                claw_angle = CLOSE_ANGLE - ((openness / 100) * (CLOSE_ANGLE - OPEN_ANGLE))
                claw_angle = max(MIN_SAFE_ANGLE, min(claw_angle, MAX_CURRENT_ANGLE))
                
                print(f"Claw: {claw_angle:.1f}° ({'Active' if claw_active else 'Inactive'}) | X: {x_angle:.1f}° | Y: {y_angle:.1f}°")
                
                # Move servos in parallel
                await asyncio.gather(
                    smooth_move(claw_servo, claw_angle),
                    smooth_move(x_servo, x_angle),
                    smooth_move(y_servo, y_angle)
                )
                
        except Exception as e:
            print(f"Error: {e}")

async def main():
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        print("WebSocket server started")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())