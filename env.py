from constants import MAX_LEN, char2idx, idx2char, BOS, EOS, idx2char

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