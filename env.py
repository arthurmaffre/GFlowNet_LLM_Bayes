from constants import MAX_LEN, char2idx, idx2char, BOS, EOS

class AddSeqEnv:
    def __init__(self, max_len=MAX_LEN):
        self.max_len = max_len

    def reset(self):
        self.state = [char2idx[BOS]]
        return self.state[:]

    def step(self, action):
        self.state.append(action)
        done = (action == char2idx[EOS]) or (len(self.state) >= self.max_len)
        reward = 0.0
        if done:
            seq_str = ''.join(idx2char[t] for t in self.state[1:-1] if t != char2idx[EOS])
            if self.preserves_causal_order(seq_str):
                reward = 1.0
            else:
                reward = 0.01
        return self.state[:], reward, done

    def preserves_causal_order(self, s):
        if '+' not in s or '=' not in s:
            return False
        parts = s.split('+')
        if len(parts) != 2:
            return False
        num1 = parts[0].strip()
        rest = parts[1].split('=')[0].strip()
        return num1.isdigit() and rest.isdigit()

if __name__ == "__main__":
    print("Testing AddSeqEnv...")

    env = AddSeqEnv()

    # Test 1: Valid sequence "1 + 2 ="
    print("\nTest 1: Building valid sequence '1 + 2 ='")
    state = env.reset()
    print(f"Reset state: {[idx2char.get(i, '<unknown>') for i in state]}")

    actions = [char2idx['1'], char2idx[' '], char2idx['+'], char2idx[' '], char2idx['2'], char2idx[' '], char2idx['='], char2idx[EOS]]
    total_reward = 0.0
    for action in actions:
        state, reward, done = env.step(action)
        char = idx2char.get(action, '<unknown>')
        seq_str = ''.join(idx2char.get(t, '<unknown>') for t in state[1:])
        print(f"Action: '{char}' (idx: {action}), State: {seq_str}, Reward: {reward}, Done: {done}")
        total_reward += reward
        if done:
            break

    if total_reward == 1.0:
        print("✓ Test 1 passed: Reward 1.0 for valid sequence.")
    else:
        print("✗ Test 1 failed: Unexpected reward.")

    # Test 2: Invalid sequence "1 2 + ="
    print("\nTest 2: Building invalid sequence '1 2 + =' (causal order violated)")
    env = AddSeqEnv()  # New instance
    state = env.reset()
    print(f"Reset state: {[idx2char.get(i, '<unknown>') for i in state]}")

    actions = [char2idx['1'], char2idx[' '], char2idx['2'], char2idx[' '], char2idx['+'], char2idx[' '], char2idx['='], char2idx[EOS]]
    total_reward = 0.0
    for action in actions:
        state, reward, done = env.step(action)
        char = idx2char.get(action, '<unknown>')
        seq_str = ''.join(idx2char.get(t, '<unknown>') for t in state[1:])
        print(f"Action: '{char}' (idx: {action}), State: {seq_str}, Reward: {reward}, Done: {done}")
        total_reward += reward
        if done:
            break

    if total_reward == 0.01:
        print("✓ Test 2 passed: Reward 0.01 for invalid sequence.")
    else:
        print("✗ Test 2 failed: Unexpected reward.")

    # Test 3: Max length reached without EOS
    print("\nTest 3: Reaching max length without EOS")
    env = AddSeqEnv()
    state = env.reset()
    total_reward = 0.0
    for i in range(MAX_LEN - 1):  # Fill up to max_len
        action = char2idx['1']  # Arbitrary action
        state, reward, done = env.step(action)
        total_reward += reward
        if done:
            seq_str = ''.join(idx2char.get(t, '<unknown>') for t in state[1:])
            print(f"Final state: {seq_str}, Total Reward: {total_reward}, Done: {done}")
            break

    if done and len(state) == MAX_LEN:
        print("✓ Test 3 passed: Done at max length.")
    else:
        print("✗ Test 3 failed: Did not reach done at max length.")

    # Test 4: Missing '=' or '+'
    print("\nTest 4: Sequence without '=' (should be invalid)")
    env = AddSeqEnv()
    state = env.reset()
    actions = [char2idx['1'], char2idx[' '], char2idx['+'], char2idx[' '], char2idx['2'], char2idx[EOS]]
    total_reward = 0.0
    for action in actions:
        state, reward, done = env.step(action)
        total_reward += reward

    if total_reward == 0.01:
        print("✓ Test 4 passed: Reward 0.01 for missing '='.")
    else:
        print("✗ Test 4 failed: Unexpected reward.")

    print("\nAll tests completed.")