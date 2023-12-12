import numpy as np
import gym, gym_ple, pickle, cv2
from gym_ple import PLEEnv
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# define action constant
ACTION_FLAP = 0
ACTION_STAY = 1

class FlappyBirdCustom(gym.Wrapper):
    def __init__(self, env, rounding = 10):
      super().__init__(env)
      self.rounding = rounding

    def _discretize(self, value):
      return self.rounding * int(value / self.rounding)

    def step(self, action):
      '''
      Hàm để tính toán và trả ra custom_next_state & custom_reward
      '''
      # Reward và Internal State của môi trường
      _, reward, terminal, _ = self.env.step(action)

      # custom reward
      if reward >= 1: # nhảy qua được ống
        custom_reward = 5
      elif terminal is False:
        custom_reward = 0.5 # sống sót sau mỗi frame
      else:
        custom_reward = -1000 # gameover

      # Do thực hiện step -> ta gọi là custom_next_state
      custom_next_state = self.get_custom_state()

      # return tuple
      return custom_next_state, custom_reward, terminal

    def get_custom_state(self):
      internal_state_dict = self.env.game_state.getGameState()

      # Tính toán distance theo trục x và y
      hor_dist = internal_state_dict['next_pipe_dist_to_player']
      ver_dist = internal_state_dict['next_pipe_bottom_y'] - internal_state_dict['player_y']
      # disretize distance
      hor_dist = self._discretize(hor_dist)
      ver_dist = self._discretize(ver_dist)
      # tính toán player đang nhảy hay rơi dựa theo velocity
      is_up = 1
      if internal_state_dict['player_vel'] >= 0:
        is_up = 0
      # custom_state cho defaultdict
      # custom_state = f"{is_up}-{hor_dist}-{ver_dist}"
      custom_state = is_up,hor_dist,ver_dist

      return custom_state

def get_optimal_action(q_values, state):
  q = [q_values[(state, action)] for action in (ACTION_FLAP, ACTION_STAY)]
  if q[ACTION_FLAP] == q[ACTION_STAY]:
    return env.action_space.sample()
  return np.argmax(q)
  
def greedy(q_values, q_counters, state):
  count = [q_counters[(state, action)] for action in (ACTION_FLAP, ACTION_STAY)]
  q = [q_values[(state, action)] for action in (ACTION_FLAP, ACTION_STAY)]
  if count[ACTION_FLAP] > 2 and count[ACTION_STAY] > 2:
    return np.argmax(q)
  if count[ACTION_FLAP] == count[ACTION_STAY]:
    return env.action_space.sample()
  return np.argmin(count)

def epsilon_greedy(q_values, state, epsilon):
  if np.random.rand() < epsilon:
    return env.action_space.sample()
  return get_optimal_action(q_values, state)
  
def update_q(q_values, state, action, next_state, reward, alpha, gamma):
  current_value = q_values[(state, action)]
  max_value = max([q_values[(next_state, action)] for action in (ACTION_FLAP, ACTION_STAY)])
  q_values[(state, action)] = current_value + alpha * (reward + gamma * max_value - current_value)

def display_frame(frame, text, title):
  frame = np.ascontiguousarray(frame)
  cv2.putText(frame, text, org=(0, 15), thickness=2,
              fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(0, 0, 255))
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  cv2.imshow(title, frame)

def train(env, episodes, epsilon_min, epsilon_decay_rate, max_steps=1000, gamma=1, alpha=.9, display=False):
  q_values = defaultdict(float)
  steps = []
  pipes = []
  epsilon = 1
  for i in range(episodes):
    env.reset()
    state = env.get_custom_state()
    step_count = 0
    pipe = 0
    while True:
      action = epsilon_greedy(q_values, state, epsilon)
      next_state, reward, terminal = env.step(action)
      update_q(q_values, state, action, next_state, reward, alpha, gamma)
      state = next_state
      step_count += 1

      if reward == 5:
        pipe += 1

      if display and i % 10 == 0:
        display_frame(env.render(mode = 'rgb_array'), f'EPISODE {i} PIPE: {pipe}', 'Training')
        cv2.waitKey(1)

      if terminal or step_count > max_steps:
        break

    steps.append(step_count)
    pipes.append(pipe)

    if epsilon > epsilon_min:
      epsilon *= epsilon_decay_rate

    # epsilon = max(epsilon*epsilon_decay_rate, epsilon_min)

    if i % 100 == 99:
      print(f"Episode {i+1} - Epsilon {epsilon}")
      print(f"    - Step          : {np.mean(steps)}")
      print(f"    - Pipe          : {np.mean(pipes)}")
  
  if display:
    cv2.destroyAllWindows()

  return q_values

def train_greedy(env, episodes, max_steps=1000, gamma=1, alpha=.9, display=False):
  q_values = defaultdict(float)
  q_counters = defaultdict(float)
  steps = []
  pipes = []
  
  env.reset()
  state = env.get_custom_state()
  pipe,step = 0,0
  episode = 1
  while episode <= episodes:
    action = greedy(q_values, q_counters, state)
    q_counters[(state, action)] += 1
    next_state, reward, terminal = env.step(action)
    update_q(q_values, state, action, next_state, reward, alpha, gamma)
    state = next_state
    step += 1

    if reward == 5:
      pipe += 1

    if display and episode % 10 == 0:
      display_frame(env.render(mode = 'rgb_array'), f'EPISODE {episode} PIPE: {pipe}', 'Training')
      cv2.waitKey(1)

    if terminal:
      steps.append(step)
      pipes.append(pipe)
      pipe,step = 0,0
      if episode % 100 == 0:
        print(f"Episode {episode}")
        print(f"    - Step          : {np.mean(steps)}")
        print(f"    - Pipe          : {np.mean(pipes)}")
      episode += 1
      env.reset()

  if display:
    cv2.destroyAllWindows()

  return q_values

def test(env, q_values, episodes=None, display=True):
  env.reset()
  state = env.get_custom_state()
  pipe = 0
  episode = 1
  while True:
    action = get_optimal_action(q_values, state)
    state, reward, terminal = env.step(action)
    if reward == 5:
      pipe += 1
    if display:
      display_frame(env.render(mode = 'rgb_array'), f'EPISODE {episode} PIPE: {pipe}', 'Testing')

    if terminal:
      if pipe > 0:print(f'Episode {episode} : {pipe}')
      pipe = 0
      episode += 1
      env.reset()
      if episodes and episode == episodes+1:
        break

    if cv2.waitKey(20) & 0xFF == ord('q'):
      break

  if display:
    cv2.destroyAllWindows()

env = FlappyBirdCustom(gym.make('FlappyBird-v0'), rounding = 10)

# q_values = train(env, episodes=50, epsilon_min=.001, epsilon_decay_rate=.99, max_steps=1000, display=False)
q_values = train_greedy(env, episodes=100, max_steps=1000, display=True)
# with open('q.pkl', 'wb') as f:
#   pickle.dump(q_values, f)

# with open('q.pkl', 'rb') as f:
#   q_values = pickle.load(f)
test(env, q_values, episodes=None, display=True)