import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from PIL import Image
import numpy as np
import mss
import random
import time
from percentages import get_percents
from inputfuncs import press, set_stick, release_everythang, release, p_and_r
from collections import deque

class Agent:
    def __init__(self, file=None, batch_size=None):
        self.memory = deque(maxlen=600) # maxlen is number of frames to remember
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.buttons = ["A", "B", "Z", "L", "NOOP"]
        self.sticks = [(1.0, 1.0), (1.0, 0.5), (1.0, 0.0), (0.5, 0.0),
                       (0.0, 0.0), (0.0, 0.5), (0.0, 1.0), (0.5, 1.0), (0.5, 0.5)]
        self.action_size = len(self.buttons) + len(self.sticks)
        self.model = self.build_model(file)
        self.batch_size = batch_size
        self.last = 4

    def build_model(self, file):
        r_w, r_h = 256, 192

        model = Sequential()
        model.add(Conv2D(10, kernel_size=(7,7), strides=2, activation='relu', input_shape=(r_w, r_h, 1)))
        model.add(MaxPooling2D(pool_size=(2,2), strides=2))
        model.add(Conv2D(20, kernel_size=(5,5), activation='relu'))
        model.add(Conv2D(24, kernel_size=(3,3), activation='relu'))
        model.add(Conv2D(24, kernel_size=(3,3), activation='relu'))
        model.add(Conv2D(24, kernel_size=(3,3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))

        if file != None:
           model.load_weights(file)

        model.compile(loss='mse', optimizer='adam')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        # Pause the game while training
        release_everythang(self.buttons)
        p_and_r("START")

        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target_butt = reward
            target_stick = reward

            if not done:
                target_butt = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0][:5])
                target_stick = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0][5:])
                    # have to change this to be not amax
            
            target_f = self.model.predict(state)
            target_f[0][np.argmax(action[:5])] = target_butt
            target_f[0][np.argmax(action[5:])+5] = target_stick
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Unpause the game after training
        p_and_r("START")

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            button_act = random.randrange(5)
            stick_act = random.randrange(9)
            act_values = np.zeros((14))
            act_values[button_act] = 1
            act_values[stick_act+5] = 1
        else:
            act_values = self.model.predict(state)[0]
            button_act = np.argmax(act_values[:5])
            stick_act = np.argmax(act_values[5:])
        
        release(self.buttons[self.last])
        self.last = button_act
        if button_act == 4:
            pass
        else:
            press(self.buttons[button_act])
        set_stick(self.sticks[stick_act])
        return act_values

def get_screen(sct, s_w, s_h):
    mon = {"top": s_h-480, "left": s_w-640, "width": 640, "height": 480}

    while True:
        im = sct.grab(mon)
        im = Image.frombytes("RGB", im.size, im.bgra, "raw", "BGRX")
        im = np.asarray(im.convert('L').resize((256,192)))
        im = np.array(im).astype('float32')
        im /= 255
        im = im.reshape(1, 256, 192, 1)
        yield(im)

def get_rewards(sct, s_w, s_h):
    agent_health = get_percents(sct, s_w, s_h, 1)
    enemy_health = get_percents(sct, s_w, s_h, 0)

    prev_agent_health = next(agent_health)
    prev_enemy_health = next(enemy_health)
    while True:
        cur_agent_health = next(agent_health)
        cur_enemy_health = next(enemy_health)
        diff_agent = cur_agent_health - prev_agent_health
        diff_enemy = cur_enemy_health - prev_enemy_health
        if prev_enemy_health == 1000:
            diff_enemy = 0
        if prev_agent_health == 1000:
            diff_agent = 0

        yield(diff_enemy - diff_agent)
        prev_agent_health = cur_agent_health
        prev_enemy_health = cur_enemy_health

def input_action(action):
    buttons = ["A", "B", "L", "Z", "RELEASE",
               "UP", "UPLEFT", "LEFT", "DOWNLEFT",
               "DOWN", "DOWNRIGHT", "RIGHT",
               "UPRIGHT", "NOMOVE"]

    if action < 5:
        press(buttons[action])
    

# main runs ONE episode for now
def main():
    sct = mss.mss()
    s_w = 1920
    s_h = 1080
    state_gen = get_screen(sct, s_w, s_h)
    state = next(state_gen)
    reward_gen = get_rewards(sct, s_w, s_h)
    batch_size = 60

    agent = Agent(file="agentweights-4.343478260869565.hdf5",
                  batch_size=batch_size)

    start_time = time.time()
    while True:
        sum_reward = 0
        n_reward = 0
        while time.time() - start_time < 600:
            for i in range(batch_size):
                action = agent.act(state)
                next_state = next(state_gen)
                reward = next(reward_gen)
                sum_reward += reward
                n_reward += 1
                agent.remember(state, action, reward, next_state, False)
                state = next_state
            agent.replay()
        start_time = time.time()
        avg_reward = sum_reward / n_reward
        print("Average reward: ", avg_reward)
        agent.model.save_weights("agentweights{}.hdf5".format(avg_reward))

def run(model_name):
    agent = Agent(file=model_name)
    state_gen = get_screen(mss.mss(), 1920, 1080)
    while True:
        action = agent.act(next(state_gen))

main()
#run('agentweights.hdf5')


