import numpy as np
import matplotlib.pyplot as plt

# Example metrics
##sac vs. td3
# A = np.array([0,1,1,1,1,1]) #np.array([0,1,1,1,1,0])
# B = np.array([1,0,0,0,0,0])

##sac vs ddpg
# A = np.array([1,1,1,1,1,1]) #np.array([1,1,1,1,1,0])
# B = np.array([0,0,0,0,0,0])

#td3 vs ddpg
A = np.array([1,1,1,0,1,1])
B = np.array([0,0,0,1,0,0])

# Compute observed wins (ignoring ties)
mask = A != B
observed_wins = np.sum(A[mask] > B[mask])
n_metrics = np.sum(mask)
print('observed_wins: ', observed_wins)
print('n_metrics: ', n_metrics)
# xxx

# Estimate probability that A beats B
prob_A_beats_B = observed_wins / n_metrics
print('prob_A_beats_B: ', prob_A_beats_B)

# Permutation test
n_perm = 10000
perm_wins = []

for _ in range(n_perm):
    swap = np.random.rand(len(A[mask])) < 0.5
    perm_A = np.where(swap, B[mask], A[mask])
    perm_B = np.where(swap, A[mask], B[mask])
    perm_wins.append(np.sum(perm_A > perm_B))

perm_wins = np.array(perm_wins)
p_value = np.mean(perm_wins >= observed_wins)
print('p_value: ', p_value)
# xxx

# Print results
print(f"Observed wins A: {observed_wins} / {n_metrics} non-tie metrics")
print(f"Estimated probability A beats B: {prob_A_beats_B:.3f}")
print(f"Permutation-based p-value (one-sided): {p_value:.4f}")
xxx

# Visualize
plt.hist(perm_wins, bins=np.arange(n_metrics+2)-0.5, color='skyblue', edgecolor='black')
plt.axvline(observed_wins, color='red', linestyle='dashed', linewidth=2, label='Observed wins')
plt.xlabel('Number of wins for A')
plt.ylabel('Frequency in permutations')
plt.title(f'Sign Test with Permutation Resampling\nP(A>B)={prob_A_beats_B:.2f}, p-value={p_value:.4f}')
plt.legend()
plt.show()
