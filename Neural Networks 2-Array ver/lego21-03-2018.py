



from ev3dev import *
from ev3dev.auto import *
import time

#Come back and check to see all of them are in right port
ir = Sensor('in1:i2c8') # It's an old sensor, so it needs i2c8
gyro = GyroSensor('in2')
steer = [LargeMotor('outA'),LargeMotor('outB')]
hold = LargeMotor('outD')
ultra = UltrasonicSensor('in3')
co = ColorSensor('in4')
kmotor = LargeMotor('outC')
green=[0,0,0]


def initalizeNorth():
    gyro.mode = "GYRO-RATE"
    gyro.mode = "GYRO-ANG"


def lookForBall(ro):
    # If ball's in sight, go get it, else spin. On finding it, realign.
    # NOTE: ADD OBJECT DETECTION
    global SENSOR_MODE
    while True:
       # SENSOR_MODE = setcolorsensor(SENSOR_MODE, "down")
        ro, gotBall = hasBall(ro)
        irValue = getIR()
        if irValue != 0 and not isWhite():
            ro.irValues[ro.irPointer] = irValue
            ro.irPointer += 1
            if ro.irPointer >= STORAGE_LENGTH:
                ro.irPointer = 0
            angle = (irValue-5)*0.25
            move(angle)
        elif isWhite():
            print("Saw line.")
            # Back away a bit, then spin.
            move(0,-1)
            time.sleep(0.75)
            spin(1)
            time.sleep(0.75)
        else:
            if mean(ro.irValues) > 5:
                move(1)
            else:
                move(-1)
        hold.run_direct(duty_cycle_sp=95) #####
        # Check if it's found:
        if gotBall:
            return "realign", ro
        ul=ultra.value()
        if ul<CLOSENESS_THRESHOLD:  # if when moving to the sideline it gets close to something...
            print("theres a robot Run away")
            return "retreat", ro #it must be a robot so retreat

def mainFunc(ro):
    while True:
        #TODO: Check
        if btn.pressed:
            initalizeNorth()
        lookForBall(ro)


    pass

class RobotObject():
    def __init__(self):
        self.irValues = [5] * STORAGE_LENGTH
        self.irPointer = 0
        self.holdThreshold = 0
        self.holdValues = [1000] * HOLD_SR_LEN
        self.holdPointer = 0

if __name__ == '__main__':
    functions = {"moveToGoal": moveToGoal, "retreat": retreat,
                 "shoot": shoot, "lookForBall": lookForBall,
                 "realign": realign, "goalsearch": goalsearch}
    ir.mode = "AC-ALL"
    state = "lookForBall"
    print("GO!")
    holdingSR = []



