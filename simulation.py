import random
import math
import time
import threading
import pygame
import sys
import os

# These are the default signal timings
defaultRed = 150
defaultYellow = 3
defaultGreen = 15
defaultMinimum = 5
defaultMaximum = 30

# Keeping track of our traffic signals
signals = []
noOfSignals = 4
simTime = 300       # How long the simulation runs
timeElapsed = 0

# Which signal is green right now?
currentGreen = 0   
nextGreen = (currentGreen+1)%noOfSignals
currentYellow = 0   # Yellow light on or off?

# How long different vehicles take to cross
carTime = 2
bikeTime = 1
rickshawTime = 2.25 
busTime = 2.5
truckTime = 2.5

# Counting vehicles at each signal
noOfCars = 0
noOfBikes = 0
noOfBuses = 0
noOfTrucks = 0
noOfRickshaws = 0
noOfLanes = 2

# When to detect vehicles (during red light)
detectionTime = 5

# How fast they go
speeds = {'car':2.25, 'bus':1.8, 'truck':1.8, 'rickshaw':2, 'bike':2.5, 'ambulance':3.5}

# Where vehicles start from
x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}    
y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}

# Storing all our vehicles
vehicles = {'right': {0:[], 1:[], 2:[], 'crossed':0}, 'down': {0:[], 1:[], 2:[], 'crossed':0}, 'left': {0:[], 1:[], 2:[], 'crossed':0}, 'up': {0:[], 1:[], 2:[], 'crossed':0}}
vehicleTypes = {0:'car', 1:'bus', 2:'truck', 3:'rickshaw', 4:'bike', 5:'ambulance'}
directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}

# Vehicle weights for pressure calculation - how "heavy" each vehicle type is
vehicle_weights = {
    'bike': 0.5,      
    'car': 1.0,       
    'rickshaw': 1.2,  
    'bus': 2.5,       
    'truck': 3.0,
    'ambulance': 4.0      # Ambulance has highest weight for emergency priority
}

# Where to put signals, timers, and counters on screen
signalCoods = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]
vehicleCountCoods = [(480,210),(880,210),(880,550),(480,550)]
vehicleCountTexts = ["0", "0", "0", "0"]
# New coordinates for pressure display
pressureCoods = [(480,190),(880,190),(880,530),(480,530)]
pressureTexts = ["0.0", "0.0", "0.0", "0.0"]
# Adjusted pressure display (for starvation prevention visualization)
adjustedPressureCoods = [(480,170),(880,170),(880,510),(480,510)]
adjustedPressureTexts = ["0.0", "0.0", "0.0", "0.0"]

# Stop lines for each direction
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
stops = {'right': [580,580,580], 'down': [320,320,320], 'left': [810,810,810], 'up': [545,545,545]}

# Middle points for turning
mid = {'right': {'x':705, 'y':445}, 'down': {'x':695, 'y':450}, 'left': {'x':695, 'y':425}, 'up': {'x':695, 'y':400}}
rotationAngle = 3

# Spacing between vehicles
gap = 15    # When stopped
gap2 = 15   # When moving

# Emergency and pause states - IMPROVED
emergencyMode = False
emergencyQueue = []  # Queue to track multiple ambulances
activeEmergencyDirection = -1
emergencyGreenTime = 15  # Time given to emergency direction
savedSignalStates = []   # Save signal states before emergency
interruptedGreen = -1    # Track which signal was interrupted by emergency
paused = False           # Pause state
signal_lock = threading.Lock()  # Thread safety
is_processing_emergency_transition = False  # Prevent recursive emergency transitions

# Performance Metrics
class PerformanceMetrics:
    def __init__(self):
        self.total_wait_time = 0
        self.vehicles_processed = 0
        self.avg_wait_per_vehicle = 0
        self.emergency_response_times = []
        self.signal_switches = 0
        self.pressure_history = {i: [] for i in range(4)}
    
    def update(self):
        """Calculate real-time metrics"""
        total_waiting = sum(
            len([v for v in vehicles[directionNumbers[i]][lane] if v.crossed == 0])
            for i in range(noOfSignals)
            for lane in range(3)
        )
        self.total_wait_time += total_waiting
        
        total_crossed = sum(vehicles[d]['crossed'] for d in directionNumbers.values())
        if total_crossed > 0:
            self.avg_wait_per_vehicle = self.total_wait_time / total_crossed
            self.vehicles_processed = total_crossed
    
    def record_pressure(self, direction_idx, pressure):
        """Record pressure for analysis"""
        self.pressure_history[direction_idx].append(pressure)

metrics = PerformanceMetrics()

# Q-Learning Traffic Controller (Optional - can be enabled)
class QLearningTrafficController:
    def __init__(self, n_signals=4, n_states=100, n_actions=4):
        self.q_table = {}  # Use dictionary for sparse state space
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        self.enabled = False  # Start disabled, can be toggled
    
    def get_state(self):
        """Convert current traffic to state index"""
        pressures = [calculate_pressure(directionNumbers[i]) for i in range(noOfSignals)]
        # Discretize pressures into state bins (0-9 for each direction)
        state = tuple(min(int(p), 9) for p in pressures)
        return state
    
    def choose_action(self, state):
        """Choose next signal using epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Explore
        
        if state not in self.q_table:
            self.q_table[state] = [0.0] * noOfSignals
        
        return self.q_table[state].index(max(self.q_table[state]))  # Exploit
    
    def update(self, state, action, reward, next_state):
        """Update Q-table"""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * noOfSignals
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * noOfSignals
        
        best_next = max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (
            reward + self.discount_factor * best_next - self.q_table[state][action]
        )
    
    def calculate_reward(self):
        """Reward = negative total waiting vehicles"""
        total_wait = sum(
            len([v for v in vehicles[directionNumbers[i]][lane] if v.crossed == 0])
            for i in range(noOfSignals)
            for lane in range(3)
        )
        return -total_wait

# Traffic Predictor using moving averages
class TrafficPredictor:
    def __init__(self, window_size=60):
        from collections import deque
        self.history = {i: deque(maxlen=window_size) for i in range(noOfSignals)}
        self.enabled = False  # Start disabled
    
    def record(self, direction_idx, pressure):
        """Record pressure measurement"""
        self.history[direction_idx].append(pressure)
    
    def predict_next(self, direction_idx, steps_ahead=5):
        """Simple moving average prediction"""
        if len(self.history[direction_idx]) < 3:
            return calculate_pressure(directionNumbers[direction_idx])
        
        recent = list(self.history[direction_idx])[-10:]
        if len(recent) < 2:
            return recent[-1] if recent else 0
        
        trend = (recent[-1] - recent[0]) / len(recent)
        prediction = recent[-1] + trend * steps_ahead
        return max(0, prediction)

# Initialize ML components (disabled by default)
q_learner = QLearningTrafficController()
predictor = TrafficPredictor()

pygame.init()
simulation = pygame.sprite.Group()

class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0
        self.waitTime = 0  # Track waiting time for starvation prevention
        self.lastGreenTime = 0  # When was last green
        
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn, is_emergency=False):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        self.is_emergency = is_emergency
        self.emergency_handled = False  # Track if this ambulance triggered emergency
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        
        # Load appropriate image based on vehicle type
        if is_emergency:
            path = "images/" + direction + "/ambulance.png"
        else:
            path = "images/" + direction + "/" + vehicleClass + ".png"
            
        # Check if file exists, use default if not
        try:
            self.originalImage = pygame.image.load(path)
        except:
            # Fallback to car image if ambulance image doesn't exist
            if is_emergency:
                path = "images/" + direction + "/car.png"
                self.originalImage = pygame.image.load(path)
                # Color it red to indicate ambulance
                self.originalImage.fill((255, 0, 0, 128), special_flags=pygame.BLEND_RGBA_MULT)
            else:
                self.originalImage = pygame.image.load(path)
                
        self.currentImage = self.originalImage.copy()
    
        # Setting up where this vehicle should stop
        if(direction=='right'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().width - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap    
            x[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif(direction=='left'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().width + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] += temp
            stops[direction][lane] += temp
        elif(direction=='down'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().height - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif(direction=='up'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().height + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] += temp
            stops[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))

    def move(self):
        if paused:
            return
            
        # Check if this is an ambulance that should trigger emergency
        if self.is_emergency and not self.emergency_handled and self.crossed == 0:
            trigger_emergency(self.direction_number, self)
            self.emergency_handled = True
            
        effective_speed = self.speed
        current_gap2 = gap2
            
        # Moving vehicles based on their direction and whether they're turning
        if(self.direction=='right'):
            if(self.crossed==0 and self.x+self.currentImage.get_rect().width>stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                # If this ambulance has crossed, remove from emergency queue
                if self.is_emergency and self in emergencyQueue:
                    remove_from_emergency_queue(self)
            if(self.willTurn==1):
                if(self.crossed==0 or self.x+self.currentImage.get_rect().width<mid[self.direction]['x']):
                    if((self.x+self.currentImage.get_rect().width<=self.stop or (currentGreen==0 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - current_gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.x += effective_speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 2
                        self.y += 1.8
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - current_gap2) or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - current_gap2)):
                            self.y += effective_speed
            else: 
                if((self.x+self.currentImage.get_rect().width<=self.stop or self.crossed == 1 or (currentGreen==0 and currentYellow==0)) and (self.index==0 or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - current_gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.x += effective_speed

        elif(self.direction=='down'):
            if(self.crossed==0 and self.y+self.currentImage.get_rect().height>stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if self.is_emergency and self in emergencyQueue:
                    remove_from_emergency_queue(self)
            if(self.willTurn==1):
                if(self.crossed==0 or self.y+self.currentImage.get_rect().height<mid[self.direction]['y']):
                    if((self.y+self.currentImage.get_rect().height<=self.stop or (currentGreen==1 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - current_gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.y += effective_speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 2.5
                        self.y += 2
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + current_gap2) or self.y<(vehicles[self.direction][self.lane][self.index-1].y - current_gap2)):
                            self.x -= effective_speed
            else: 
                if((self.y+self.currentImage.get_rect().height<=self.stop or self.crossed == 1 or (currentGreen==1 and currentYellow==0)) and (self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - current_gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.y += effective_speed
            
        elif(self.direction=='left'):
            if(self.crossed==0 and self.x<stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if self.is_emergency and self in emergencyQueue:
                    remove_from_emergency_queue(self)
            if(self.willTurn==1):
                if(self.crossed==0 or self.x>mid[self.direction]['x']):
                    if((self.x>=self.stop or (currentGreen==2 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + current_gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.x -= effective_speed
                else: 
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 1.8
                        self.y -= 2.5
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + current_gap2) or self.x>(vehicles[self.direction][self.lane][self.index-1].x + current_gap2)):
                            self.y -= effective_speed
            else: 
                if((self.x>=self.stop or self.crossed == 1 or (currentGreen==2 and currentYellow==0)) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + current_gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.x -= effective_speed

        elif(self.direction=='up'):
            if(self.crossed==0 and self.y<stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if self.is_emergency and self in emergencyQueue:
                    remove_from_emergency_queue(self)
            if(self.willTurn==1):
                if(self.crossed==0 or self.y>mid[self.direction]['y']):
                    if((self.y>=self.stop or (currentGreen==3 and currentYellow==0) or self.crossed == 1) and (self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + current_gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):
                        self.y -= effective_speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 1
                        self.y -= 1
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.x<(vehicles[self.direction][self.lane][self.index-1].x - vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width - current_gap2) or self.y>(vehicles[self.direction][self.lane][self.index-1].y + current_gap2)):
                            self.x += effective_speed
            else: 
                if((self.y>=self.stop or self.crossed == 1 or (currentGreen==3 and currentYellow==0)) and (self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + current_gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.y -= effective_speed

# Function to calculate pressure on a lane
def calculate_pressure(direction):
    total_pressure = 0.0
    
    for lane in range(3):
        for vehicle in vehicles[direction][lane]:
            if vehicle.crossed == 0:
                total_pressure += vehicle_weights.get(vehicle.vehicleClass, 1.0)
    
    return total_pressure

# Setting up our traffic signals
def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    ts1.lastGreenTime = 0  # First signal starts green
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red+ts1.yellow+ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)
    repeat()

# Calculate how long the next green light should be - IMPROVED with pressure
def setTime():
    global noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws, noOfLanes
    global carTime, busTime, truckTime, rickshawTime, bikeTime
    
    # Calculate pressure for next direction
    pressure = calculate_pressure(directionNumbers[nextGreen])
    
    # Record pressure for metrics
    metrics.record_pressure(nextGreen, pressure)
    
    # Base green time on pressure with optimized scaling factor
    # Scaling factor adjusted to match actual vehicle crossing times
    # Average crossing time per pressure unit ≈ 1.2 seconds
    # This ensures vehicles finish crossing close to when green ends
    greenTime = math.ceil(pressure * 1.2)
    
    # Apply constraints
    if greenTime < defaultMinimum:
        greenTime = defaultMinimum
    elif greenTime > defaultMaximum:
        greenTime = defaultMaximum
    
    signals[nextGreen].green = greenTime
    
    print(f'Next Green ({directionNumbers[nextGreen]}): {greenTime}s (Pressure: {pressure:.1f})')

# Intelligent signal selection with starvation prevention
def select_next_signal():
    """Select next signal based on pressure and fairness"""
    global timeElapsed
    
    pressures = []
    current_time = timeElapsed
    
    for i in range(noOfSignals):
        if i != currentGreen:
            pressure = calculate_pressure(directionNumbers[i])
            wait_time = current_time - signals[i].lastGreenTime
            
            # Only apply starvation prevention after default green time (15s)
            # This prevents premature boosting during normal operation
            if wait_time > 15:
                # EXPONENTIAL scaling for more aggressive starvation prevention
                # After 15s: multiplier starts at 1.0
                # After 30s: multiplier ≈ 1.4 (40% boost)
                # After 45s: multiplier ≈ 2.0 (100% boost)
                # After 60s: multiplier ≈ 2.8 (180% boost)
                # After 75s: multiplier ≈ 4.0 (300% boost)
                extra_wait = wait_time - 15
                starvation_multiplier = math.pow(1.05, extra_wait)  # Exponential growth
                adjusted_pressure = pressure * starvation_multiplier
            else:
                # Within normal wait time, no adjustment needed
                adjusted_pressure = pressure
            
            pressures.append((i, adjusted_pressure, pressure, wait_time))
    
    # Sort by adjusted pressure (descending)
    pressures.sort(key=lambda x: x[1], reverse=True)
    
    if pressures:
        selected = pressures[0]
        print(f"Signal selection: {directionNumbers[selected[0]]} - Raw: {selected[2]:.1f}, Wait: {selected[3]}s, Adjusted: {selected[1]:.1f}")
        return selected[0]
    
    # Fallback to round-robin
    return (currentGreen + 1) % noOfSignals

# Internal emergency trigger (assumes lock is already held)
def _trigger_emergency_internal(direction_number, ambulance_vehicle):
    global emergencyMode, activeEmergencyDirection, savedSignalStates, currentGreen, currentYellow, nextGreen, interruptedGreen
    
    # Add ambulance to queue if not already there
    if ambulance_vehicle not in emergencyQueue:
        emergencyQueue.append(ambulance_vehicle)
        print(f"Ambulance added to emergency queue from {directionNumbers[direction_number]}. Queue size: {len(emergencyQueue)}")
    
    # If already in emergency mode for this direction, just extend
    if emergencyMode and activeEmergencyDirection == direction_number:
        print(f"Extending emergency mode for {directionNumbers[direction_number]}")
        return
    
    # If emergency mode active for different direction, queue this one
    if emergencyMode and activeEmergencyDirection != direction_number:
        print(f"Emergency already active for {directionNumbers[activeEmergencyDirection]}. Queueing {directionNumbers[direction_number]}")
        return
    
    # Start new emergency mode
    emergencyMode = True
    activeEmergencyDirection = direction_number
    
    # Save which signal was interrupted
    interruptedGreen = currentGreen
    print(f"EMERGENCY MODE ACTIVATED for {directionNumbers[direction_number]}")
    print(f"Interrupted signal: {directionNumbers[interruptedGreen]} (will resume after emergency)")
    
    # Save current signal states
    savedSignalStates = []
    for i in range(noOfSignals):
        savedSignalStates.append({
            'red': signals[i].red,
            'yellow': signals[i].yellow,
            'green': signals[i].green
        })
    
    # Set emergency direction to green
    currentGreen = direction_number
    currentYellow = 0
    nextGreen = (currentGreen + 1) % noOfSignals
    
    # Clear all signals first
    for i in range(noOfSignals):
        signals[i].green = 0
        signals[i].yellow = 0
        signals[i].red = 0
    
    # Set emergency lane to green
    signals[direction_number].green = emergencyGreenTime
    signals[direction_number].yellow = defaultYellow
    signals[direction_number].red = 0
    
    # Update lastGreenTime for emergency signal
    signals[direction_number].lastGreenTime = timeElapsed
    
    # All others stay red
    for i in range(noOfSignals):
        if i != direction_number:
            signals[i].red = emergencyGreenTime + defaultYellow

# IMPROVED: Handle multiple ambulances with queue
def trigger_emergency(direction_number, ambulance_vehicle):
    with signal_lock:
        _trigger_emergency_internal(direction_number, ambulance_vehicle)

# Remove ambulance from queue when it crosses
def remove_from_emergency_queue(ambulance_vehicle):
    global emergencyQueue, is_processing_emergency_transition
    
    should_end_emergency = False
    
    with signal_lock:
        if ambulance_vehicle in emergencyQueue:
            emergencyQueue.remove(ambulance_vehicle)
            print(f"Ambulance crossed. Remaining in queue: {len(emergencyQueue)}")
            
            # If no more ambulances in queue, check if we should end emergency
            if len(emergencyQueue) == 0 and not is_processing_emergency_transition:
                # Check if there are any other uncrossed emergency vehicles
                has_uncrossed = False
                for lane in range(3):
                    for v in vehicles[directionNumbers[activeEmergencyDirection]][lane]:
                        if v.is_emergency and v.crossed == 0 and v != ambulance_vehicle:
                            has_uncrossed = True
                            break
                    if has_uncrossed:
                        break
                
                if not has_uncrossed:
                    print("No more emergency vehicles. Will end emergency mode.")
                    should_end_emergency = True
    
    # CRITICAL: Call end_emergency OUTSIDE the lock to avoid deadlock
    if should_end_emergency:
        end_emergency()

# End emergency mode and restore normal operation
def end_emergency():
    global emergencyMode, activeEmergencyDirection, savedSignalStates, currentGreen, nextGreen, interruptedGreen, is_processing_emergency_transition
    
    with signal_lock:
        # Prevent recursive calls
        if is_processing_emergency_transition:
            print("Already processing emergency transition, skipping...")
            return
            
        if not emergencyMode:
            return
        
        is_processing_emergency_transition = True
        
        try:
            print("Ending emergency mode. Restoring normal operation.")
            
            emergencyMode = False
            prev_emergency_dir = activeEmergencyDirection
            activeEmergencyDirection = -1
            
            # Return to the interrupted signal
            if interruptedGreen != -1:
                print(f"Resuming interrupted signal: {directionNumbers[interruptedGreen]}")
                currentGreen = interruptedGreen
                nextGreen = (currentGreen + 1) % noOfSignals
                interruptedGreen = -1  # Reset
            
            # Restore saved signal states if available
            if savedSignalStates:
                for i in range(noOfSignals):
                    # Prevent negative values
                    signals[i].red = max(0, savedSignalStates[i]['red'])
                    signals[i].yellow = max(0, savedSignalStates[i]['yellow'])
                    signals[i].green = max(0, savedSignalStates[i]['green'])
                
                # If the interrupted signal had no green time left, give it default green time
                if signals[currentGreen].green == 0 and signals[currentGreen].yellow == 0:
                    print(f"Interrupted signal {directionNumbers[currentGreen]} had completed. Giving new green cycle.")
                    signals[currentGreen].green = defaultGreen
                    signals[currentGreen].yellow = defaultYellow
                    signals[currentGreen].red = 0
                    
                    # Recalculate red times for other signals
                    for i in range(noOfSignals):
                        if i != currentGreen:
                            signals[i].green = 0
                            signals[i].yellow = 0
                            # Calculate based on cycle
                            if i == nextGreen:
                                signals[i].red = signals[currentGreen].green + signals[currentGreen].yellow
                            else:
                                signals[i].red = defaultRed
                
                savedSignalStates = []
            else:
                # If no saved states, reset to defaults with interrupted signal getting green
                for i in range(noOfSignals):
                    if i == currentGreen:
                        signals[i].green = defaultGreen
                        signals[i].yellow = defaultYellow
                        signals[i].red = 0
                    else:
                        signals[i].green = 0
                        signals[i].yellow = 0
                        if i == nextGreen:
                            signals[i].red = signals[currentGreen].green + signals[currentGreen].yellow
                        else:
                            signals[i].red = defaultRed
            
            # Check if there are more ambulances waiting (prepare data while holding lock)
            next_ambulance_to_process = None
            if len(emergencyQueue) > 0:
                next_ambulance = emergencyQueue[0]
                
                # Validate the ambulance hasn't crossed yet
                if hasattr(next_ambulance, 'crossed') and next_ambulance.crossed == 0:
                    next_ambulance_to_process = (next_ambulance.direction_number, next_ambulance)
                    print(f"Will process next ambulance in queue from {directionNumbers[next_ambulance.direction_number]}")
                else:
                    # Ambulance already crossed, remove it and check again
                    print(f"Next ambulance in queue already crossed, removing...")
                    emergencyQueue.pop(0)
                    # Check if there are more
                    if len(emergencyQueue) > 0:
                        next_ambulance = emergencyQueue[0]
                        if hasattr(next_ambulance, 'crossed') and next_ambulance.crossed == 0:
                            next_ambulance_to_process = (next_ambulance.direction_number, next_ambulance)
                            print(f"Will process next valid ambulance from {directionNumbers[next_ambulance.direction_number]}")
        
        finally:
            # Always reset the transition flag
            is_processing_emergency_transition = False
    
    # CRITICAL: Trigger next emergency OUTSIDE the lock to avoid deadlock
    if next_ambulance_to_process:
        trigger_emergency(next_ambulance_to_process[0], next_ambulance_to_process[1])

# Function to toggle pause state
def toggle_pause():
    global paused
    
    paused = not paused
    
    if paused:
        print("Simulation PAUSED")
    else:
        print("Simulation RESUMED")

# Main loop for signal switching - FIXED
# Updated repeat function using a state-machine approach to prevent sticking
def repeat():
    global currentGreen, currentYellow, nextGreen, emergencyMode
    
    while True:
        if not paused:
            # 1. GREEN PHASE
            # We use a loop that can be interrupted by emergencyMode
            while signals[currentGreen].green > 0 and not paused:
                if emergencyMode and activeEmergencyDirection != currentGreen:
                    break # Exit this loop if an emergency starts elsewhere
                
                printStatus()
                updateValues()
                
                # Dynamic time calculation for the NEXT signal
                if signals[currentGreen].green == detectionTime:
                    thread = threading.Thread(name="detection", target=setTime, args=())
                    thread.daemon = True
                    thread.start()
                
                time.sleep(1)

            if paused: continue

            # 2. TRANSITION TO YELLOW (If not an active emergency)
            if not emergencyMode or signals[currentGreen].green == 0:
                currentYellow = 1
                # Reset stop lines for the signal that just finished
                for i in range(3):
                    stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
                    for vehicle in vehicles[directionNumbers[currentGreen]][i]:
                        vehicle.stop = defaultStop[directionNumbers[currentGreen]]

                # YELLOW PHASE
                while signals[currentGreen].yellow > 0 and not paused:
                    if emergencyMode: break # Allow emergency to override yellow if needed
                    printStatus()
                    updateValues()
                    time.sleep(1)
                
                currentYellow = 0

            # 3. SWITCH TO NEXT SIGNAL
            should_end_emergency = False  # Initialize before lock
            with signal_lock:
                if not emergencyMode:
                    # Normal Cycle transition
                    # Reset current signal to defaults for its next turn
                    signals[currentGreen].green = defaultGreen
                    signals[currentGreen].yellow = defaultYellow
                    signals[currentGreen].red = defaultRed
                    
                    # Use intelligent signal selection
                    currentGreen = select_next_signal()
                    nextGreen = (currentGreen + 1) % noOfSignals
                    
                    # Update last green time for starvation prevention
                    signals[currentGreen].lastGreenTime = timeElapsed
                    
                    # Track metrics
                    metrics.signal_switches += 1
                    
                    # Set the red time for the new green to 0 and calculate next red
                    signals[currentGreen].red = 0
                    red_time = signals[currentGreen].green + signals[currentGreen].yellow
                    signals[nextGreen].red = red_time
                else:
                    # If in emergency, we stay on activeEmergencyDirection
                    # but check if the queue is empty
                    if len(emergencyQueue) == 0:
                        has_uncrossed = False
                        for lane in range(3):
                            for v in vehicles[directionNumbers[activeEmergencyDirection]][lane]:
                                if v.is_emergency and v.crossed == 0:
                                    has_uncrossed = True
                                    break
                        if not has_uncrossed:
                            should_end_emergency = True
            
            # CRITICAL: Call end_emergency OUTSIDE the lock to avoid deadlock
            if should_end_emergency:
                end_emergency()
            
            if emergencyMode:
                time.sleep(0.5)
        else:
            time.sleep(0.5)



# Print status
def printStatus():                                                                                           
    for i in range(0, noOfSignals):
        if(i==currentGreen):
            if(currentYellow==0):
                print(" GREEN TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
            else:
                print("YELLOW TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
        else:
            print("   RED TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
    
    if emergencyMode:
        print(f"EMERGENCY MODE: {directionNumbers[activeEmergencyDirection]} | Queue: {len(emergencyQueue)} ambulances")
    print()

# Update timer values - FIXED to prevent negatives
def updateValues():
    with signal_lock:
        for i in range(0, noOfSignals):
            if(i==currentGreen):
                if(currentYellow==0):
                    signals[i].green = max(0, signals[i].green - 1)
                    signals[i].totalGreenTime += 1
                else:
                    signals[i].yellow = max(0, signals[i].yellow - 1)
            else:
                signals[i].red = max(0, signals[i].red - 1)
        
        # Update performance metrics
        metrics.update()

# Update pressure display
def update_pressure_display():
    global timeElapsed
    current_time = timeElapsed
    
    for i in range(noOfSignals):
        direction = directionNumbers[i]
        pressure = calculate_pressure(direction)
        pressureTexts[i] = f"{pressure:.1f}"
        
        # Calculate adjusted pressure (same exponential logic as select_next_signal)
        wait_time = current_time - signals[i].lastGreenTime
        if wait_time > 15:
            extra_wait = wait_time - 15
            starvation_multiplier = math.pow(1.05, extra_wait)  # Exponential growth
            adjusted_pressure = pressure * starvation_multiplier
        else:
            adjusted_pressure = pressure
        
        adjustedPressureTexts[i] = f"{adjusted_pressure:.1f}"

# Generate random vehicles
def generateVehicles():
    while(True):
        if not paused:
            vehicle_type = random.randint(0,4)
            
            if(vehicle_type==4):
                lane_number = 0
            else:
                lane_number = random.randint(0,1) + 1
            
            will_turn = 0
            if(lane_number==2):
                temp = random.randint(0,4)
                if(temp<=2):
                    will_turn = 1
                elif(temp>2):
                    will_turn = 0
            
            temp = random.randint(0,999)
            direction_number = 0
            a = [400,800,900,1000]
            if(temp<a[0]):
                direction_number = 0
            elif(temp<a[1]):
                direction_number = 1
            elif(temp<a[2]):
                direction_number = 2
            elif(temp<a[3]):
                direction_number = 3
            
            Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number], will_turn)
            
            update_pressure_display()
            
            # Reduced from 0.75s to 0.5s for more realistic traffic density
            time.sleep(0.5)
        else:
            time.sleep(0.1)

# Spawn ambulance - can be from any direction
def spawn_ambulance():
    # Random direction for ambulance
    direction_number = random.randint(0, 3)
    direction = directionNumbers[direction_number]
    
    lane_number = 1  # Middle lane
    will_turn = 0  # Straight path
    
    print(f"Spawning ambulance from {direction} direction")
    
    Vehicle(lane_number, 'ambulance', direction_number, direction, will_turn, is_emergency=True)
    
    update_pressure_display()

# Track simulation time
def simulationTime():
    global timeElapsed, simTime
    while(True):
        if not paused:
            timeElapsed += 1
        time.sleep(1)
        if(timeElapsed==simTime):
            totalVehicles = 0
            print('Lane-wise Vehicle Counts')
            for i in range(noOfSignals):
                print('Lane',i+1,':',vehicles[directionNumbers[i]]['crossed'])
                totalVehicles += vehicles[directionNumbers[i]]['crossed']
            print('Total vehicles passed: ',totalVehicles)
            print('Total time passed: ',timeElapsed)
            print('No. of vehicles passed per unit time: ',(float(totalVehicles)/float(timeElapsed)))
            os._exit(1)

class Main:
    thread4 = threading.Thread(name="simulationTime",target=simulationTime, args=()) 
    thread4.daemon = True
    thread4.start()

    thread2 = threading.Thread(name="initialization",target=initialize, args=())
    thread2.daemon = True
    thread2.start()

    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    blue = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (255, 255, 0)
    orange = (255, 165, 0)

    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    background = pygame.image.load('images/mod_int.png')

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("FIXED DYNAMIC TRAFFIC SYSTEM - Multi-Ambulance Support")

    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)
    small_font = pygame.font.Font(None, 25)
    large_font = pygame.font.Font(None, 50)

    thread3 = threading.Thread(name="generateVehicles",target=generateVehicles, args=())
    thread3.daemon = True
    thread3.start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    toggle_pause()
                elif event.key == pygame.K_e:
                    if not paused:
                        spawn_ambulance()
                    else:
                        print("Cannot spawn ambulance while paused")

        screen.blit(background,(0,0))
        
        # Display traffic signals
        for i in range(0,noOfSignals):
            if(i==currentGreen):
                if(currentYellow==1):
                    if(signals[i].yellow==0):
                        signals[i].signalText = "STOP"
                    else:
                        signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    if(signals[i].green==0):
                        signals[i].signalText = "SLOW"
                    else:
                        signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                if(signals[i].red<=10):
                    if(signals[i].red==0):
                        signals[i].signalText = "GO"
                    else:
                        signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])
        
        signalTexts = ["","","",""]
        
        # Display timers, counts, and pressure
        for i in range(0,noOfSignals):  
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i],signalTimerCoods[i])
            
            displayText = vehicles[directionNumbers[i]]['crossed']
            vehicleCountTexts[i] = font.render(str(displayText), True, black, white)
            screen.blit(vehicleCountTexts[i],vehicleCountCoods[i])
            
            # Display raw pressure
            pressure_text = small_font.render(f"P: {pressureTexts[i]}", True, red, white)
            screen.blit(pressure_text, pressureCoods[i])
            
            # Display adjusted pressure (with starvation prevention)
            adjusted_text = small_font.render(f"Adj: {adjustedPressureTexts[i]}", True, orange, white)
            screen.blit(adjusted_text, adjustedPressureCoods[i])
        
        timeElapsedText = font.render(("Time: "+str(timeElapsed)), True, black, white)
        screen.blit(timeElapsedText,(1100,50))
        
        # Legend
        legend_y = 100
        legend_text1 = small_font.render("Pressure Weights:", True, blue, white)
        screen.blit(legend_text1, (1100, legend_y))
        
        legend_items = [
            "Bike: 0.5",
            "Car: 1.0", 
            "Rickshaw: 1.2",
            "Bus: 2.5",
            "Truck: 3.0",
            "Ambulance: 4.0"
        ]
        
        for idx, item in enumerate(legend_items):
            item_text = small_font.render(item, True, black, white)
            screen.blit(item_text, (1100, legend_y + 25 * (idx + 1)))
        
        
        # Pause status
        if paused:
            pause_text = large_font.render("PAUSED", True, yellow, black)
            screen.blit(pause_text, (600, 50))
        
        # Emergency status (thread-safe display)
        # Capture emergency state atomically to avoid race conditions
        with signal_lock:
            display_emergency = emergencyMode
            if display_emergency and activeEmergencyDirection >= 0:
                emergency_dir = activeEmergencyDirection
                emergency_queue_size = len(emergencyQueue)
                emergency_green_time = signals[activeEmergencyDirection].green
            else:
                display_emergency = False
        
        if display_emergency:
            emergency_text = large_font.render("EMERGENCY!", True, red, black)
            screen.blit(emergency_text, (200, 100))
            
            direction_text = font.render(f"Active: {directionNumbers[emergency_dir].upper()}", True, red, white)
            screen.blit(direction_text, (200, 150))
            
            queue_text = font.render(f"Queue: {emergency_queue_size} ambulance(s)", True, orange, white)
            screen.blit(queue_text, (200, 180))
            
            timer_text = font.render(f"Green time: {emergency_green_time}s", True, green, white)
            screen.blit(timer_text, (200, 210))
        
        # Performance Metrics Display
        metrics_y = 600
        metrics_title = small_font.render("Performance Metrics:", True, blue, white)
        screen.blit(metrics_title, (50, metrics_y))
        
        avg_wait_text = small_font.render(f"Avg Wait/Vehicle: {metrics.avg_wait_per_vehicle:.2f}", True, black, white)
        screen.blit(avg_wait_text, (50, metrics_y + 25))
        
        processed_text = small_font.render(f"Vehicles Processed: {metrics.vehicles_processed}", True, black, white)
        screen.blit(processed_text, (50, metrics_y + 50))
        
        switches_text = small_font.render(f"Signal Switches: {metrics.signal_switches}", True, black, white)
        screen.blit(switches_text, (50, metrics_y + 75))

        # Move and draw vehicles
        for vehicle in simulation:  
            screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
            if not paused:
                vehicle.move()

        pygame.display.update()

Main()