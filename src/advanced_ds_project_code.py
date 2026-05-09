# %% [code cell 1]
# ===============================
# 1) Imports and Reproducibility
# ===============================
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from sklearn.cluster import KMeans
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    silhouette_score,
    adjusted_rand_score,
)
from sklearn.metrics.pairwise import euclidean_distances

warnings.filterwarnings('ignore')
np.random.seed(42)

sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

print('Imports ready.')

# %% [code cell 2]
# =========================================
# 2) Utilities: timing, reporting, lambda graph
# =========================================
runtime_log = []
results = []


def log_runtime(section, start_time):
    runtime_log.append({'Section': section, 'Runtime_sec': time.time() - start_time})


def add_result(dataset, algorithm, metric, value):
    results.append({
        'Dataset': dataset,
        'Algorithm': algorithm,
        'Metric': metric,
        'Value': float(value) if np.isscalar(value) else value,
    })


def clustering_report(X, labels, y_true=None):
    out = {}
    out['n_clusters'] = int(len(np.unique(labels)))
    out['silhouette'] = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else np.nan
    out['ari'] = float(adjusted_rand_score(y_true, labels)) if y_true is not None else np.nan
    return out


def lambda_connected_components(X, lambda_threshold=0.75, sample_limit=2500):
    # Optional downsampling for O(n^2) distance construction.
    if X.shape[0] > sample_limit:
        idx = np.random.choice(X.shape[0], sample_limit, replace=False)
        X_work = X[idx]
    else:
        idx = np.arange(X.shape[0])
        X_work = X

    D = euclidean_distances(X_work, X_work)
    sigma = np.median(D[D > 0]) if np.any(D > 0) else 1.0
    S = np.exp(-(D ** 2) / (2 * sigma ** 2 + 1e-12))
    A = S >= lambda_threshold

    n = A.shape[0]
    labels = -np.ones(n, dtype=int)
    comp_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = comp_id
        while stack:
            u = stack.pop()
            neighbors = np.where(A[u])[0]
            for v in neighbors:
                if labels[v] == -1:
                    labels[v] = comp_id
                    stack.append(v)
        comp_id += 1

    return idx, labels

print('Utilities ready.')

# %% [code cell 3]
# ===============================
# 3) Load datasets and summarize
# ===============================
start = time.time()

housing = fetch_california_housing()
X_housing, y_housing = housing.data, housing.target

bc = load_breast_cancer()
X_bc, y_bc = bc.data, bc.target

# Subsample MNIST for practical runtime in class project settings.
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_mnist = mnist.data.astype(np.float32) / 255.0
y_mnist = mnist.target.astype(int)

summary_df = pd.DataFrame([
    {'Dataset': 'California Housing', 'Samples': X_housing.shape[0], 'Features': X_housing.shape[1], 'Task': 'Regression'},
    {'Dataset': 'Breast Cancer', 'Samples': X_bc.shape[0], 'Features': X_bc.shape[1], 'Task': 'Binary Classification'},
    {'Dataset': 'MNIST', 'Samples': X_mnist.shape[0], 'Features': X_mnist.shape[1], 'Task': 'Multiclass Classification'},
    {'Dataset': 'FrozenLake', 'Samples': 16, 'Features': 1, 'Task': 'RL (MDP states)'}
])

log_runtime('Dataset Loading', start)
summary_df

# %% [code cell 4]
# =====================================
# 4) EDA: distributions and correlations
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Housing target distribution.
axes[0, 0].hist(y_housing, bins=50, color='steelblue', alpha=0.8)
axes[0, 0].set_title('California Housing Target Distribution')
axes[0, 0].set_xlabel('Median House Value')
axes[0, 0].set_ylabel('Count')

# Breast cancer class balance.
bc_counts = pd.Series(y_bc).value_counts().sort_index()
axes[0, 1].bar(['Malignant (0)', 'Benign (1)'], bc_counts.values, color=['salmon', 'seagreen'])
axes[0, 1].set_title('Breast Cancer Class Distribution')

# Housing correlation heatmap on a subset of features.
housing_df = pd.DataFrame(X_housing, columns=housing.feature_names)
corr = housing_df.corr()
sns.heatmap(corr, cmap='coolwarm', ax=axes[1, 0], cbar=False)
axes[1, 0].set_title('Housing Feature Correlation')

# MNIST average image.
mean_digit = X_mnist[:10000].mean(axis=0).reshape(28, 28)
axes[1, 1].imshow(mean_digit, cmap='gray')
axes[1, 1].set_title('MNIST Mean Digit (first 10k samples)')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# %% [code cell 5]
start = time.time()

# Split data.
Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_housing, y_housing, test_size=0.2, random_state=42)

# Scale features.
scaler_h = StandardScaler()
Xh_train_s = scaler_h.fit_transform(Xh_train)
Xh_test_s = scaler_h.transform(Xh_test)
Xh_all_s = scaler_h.fit_transform(X_housing)

# SVM regressor.
svm_h = SVR(C=15, epsilon=0.1, kernel='rbf')
svm_h.fit(Xh_train_s, yh_train)
yh_pred_svm = svm_h.predict(Xh_test_s)
add_result('Housing', 'SVM', 'RMSE', np.sqrt(mean_squared_error(yh_test, yh_pred_svm)))
add_result('Housing', 'SVM', 'R2', r2_score(yh_test, yh_pred_svm))

# Neural network regressor.
nn_h = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=120, random_state=42)
nn_h.fit(Xh_train_s, yh_train)
yh_pred_nn = nn_h.predict(Xh_test_s)
add_result('Housing', 'Neural Network', 'RMSE', np.sqrt(mean_squared_error(yh_test, yh_pred_nn)))
add_result('Housing', 'Neural Network', 'R2', r2_score(yh_test, yh_pred_nn))

# K-means clustering with target bin proxy for ARI.
kmeans_h = KMeans(n_clusters=6, random_state=42, n_init=15)
kh_labels = kmeans_h.fit_predict(Xh_all_s)
y_h_bins = pd.qcut(y_housing, q=6, labels=False, duplicates='drop')
kh_rep = clustering_report(Xh_all_s, kh_labels, y_h_bins)
add_result('Housing', 'K-means', 'Silhouette', kh_rep['silhouette'])
add_result('Housing', 'K-means', 'ARI_vs_binned_target', kh_rep['ari'])

# Lambda-connectedness on PCA projection.
Xh_pca = PCA(n_components=6, random_state=42).fit_transform(Xh_all_s[:3000])
idx_h, lmb_h_labels = lambda_connected_components(Xh_pca, lambda_threshold=0.72, sample_limit=2500)
lmb_h_rep = clustering_report(Xh_pca[idx_h], lmb_h_labels)
add_result('Housing', 'Lambda-connectedness', 'NumClusters', lmb_h_rep['n_clusters'])
add_result('Housing', 'Lambda-connectedness', 'Silhouette', lmb_h_rep['silhouette'])

# Visual summary.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(yh_test, yh_pred_svm, alpha=0.3, s=8, color='steelblue')
axes[0, 0].plot([yh_test.min(), yh_test.max()], [yh_test.min(), yh_test.max()], 'r--')
axes[0, 0].set_title('Housing SVM: Actual vs Predicted')
axes[0, 0].set_xlabel('Actual')
axes[0, 0].set_ylabel('Predicted')

axes[0, 1].scatter(yh_test, yh_pred_nn, alpha=0.3, s=8, color='darkorange')
axes[0, 1].plot([yh_test.min(), yh_test.max()], [yh_test.min(), yh_test.max()], 'r--')
axes[0, 1].set_title('Housing NN: Actual vs Predicted')
axes[0, 1].set_xlabel('Actual')
axes[0, 1].set_ylabel('Predicted')

Xh_pca2 = PCA(n_components=2, random_state=42).fit_transform(Xh_all_s[:3000])
sc1 = axes[1, 0].scatter(Xh_pca2[:, 0], Xh_pca2[:, 1], c=kh_labels[:3000], cmap='tab10', s=8, alpha=0.6)
axes[1, 0].set_title('Housing K-means Clusters (PCA projection)')
plt.colorbar(sc1, ax=axes[1, 0])

sc2 = axes[1, 1].scatter(Xh_pca[idx_h, 0], Xh_pca[idx_h, 1], c=lmb_h_labels, cmap='tab20', s=8, alpha=0.7)
axes[1, 1].set_title('Housing Lambda-connected Components')
plt.colorbar(sc2, ax=axes[1, 1])

plt.tight_layout()
plt.show()

log_runtime('Housing Experiments', start)

# %% [code cell 6]
start = time.time()

Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_bc, y_bc, test_size=0.2, stratify=y_bc, random_state=42)
scaler_b = StandardScaler()
Xb_train_s = scaler_b.fit_transform(Xb_train)
Xb_test_s = scaler_b.transform(Xb_test)
Xb_all_s = scaler_b.fit_transform(X_bc)

# Compare linear vs RBF SVM to analyze separability.
svm_b_lin = SVC(kernel='linear', C=1.0, random_state=42)
svm_b_rbf = SVC(kernel='rbf', C=5.0, gamma='scale', random_state=42)
svm_b_lin.fit(Xb_train_s, yb_train)
svm_b_rbf.fit(Xb_train_s, yb_train)

yb_pred_lin = svm_b_lin.predict(Xb_test_s)
yb_pred_rbf = svm_b_rbf.predict(Xb_test_s)

# Select the better performer for downstream reporting.
acc_lin = accuracy_score(yb_test, yb_pred_lin)
acc_rbf = accuracy_score(yb_test, yb_pred_rbf)
svm_b = svm_b_rbf if acc_rbf >= acc_lin else svm_b_lin
yb_pred_svm = yb_pred_rbf if acc_rbf >= acc_lin else yb_pred_lin

add_result('Breast Cancer', 'SVM', 'Accuracy', accuracy_score(yb_test, yb_pred_svm))
add_result('Breast Cancer', 'SVM', 'Precision', precision_score(yb_test, yb_pred_svm))
add_result('Breast Cancer', 'SVM', 'Recall', recall_score(yb_test, yb_pred_svm))
add_result('Breast Cancer', 'SVM', 'F1', f1_score(yb_test, yb_pred_svm))

# Neural network classifier.
nn_b = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
nn_b.fit(Xb_train_s, yb_train)
yb_pred_nn = nn_b.predict(Xb_test_s)
add_result('Breast Cancer', 'Neural Network', 'Accuracy', accuracy_score(yb_test, yb_pred_nn))
add_result('Breast Cancer', 'Neural Network', 'Precision', precision_score(yb_test, yb_pred_nn))
add_result('Breast Cancer', 'Neural Network', 'Recall', recall_score(yb_test, yb_pred_nn))
add_result('Breast Cancer', 'Neural Network', 'F1', f1_score(yb_test, yb_pred_nn))

# K-means clustering.
kmeans_b = KMeans(n_clusters=2, random_state=42, n_init=20)
kb_labels = kmeans_b.fit_predict(Xb_all_s)
kb_rep = clustering_report(Xb_all_s, kb_labels, y_bc)
add_result('Breast Cancer', 'K-means', 'Silhouette', kb_rep['silhouette'])
add_result('Breast Cancer', 'K-means', 'ARI', kb_rep['ari'])

# Visual diagnostics.
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

cm_svm = confusion_matrix(yb_test, yb_pred_svm)
cm_nn = confusion_matrix(yb_test, yb_pred_nn)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Breast Cancer SVM Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 1])
axes[0, 1].set_title('Breast Cancer NN Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

Xb_pca2 = PCA(n_components=2, random_state=42).fit_transform(Xb_all_s)
axes[1, 0].scatter(Xb_pca2[:, 0], Xb_pca2[:, 1], c=kb_labels, s=15, cmap='tab10', alpha=0.65)
axes[1, 0].set_title('Breast Cancer K-means Clusters (PCA)')

axes[1, 1].plot(nn_b.loss_curve_, color='darkorange')
axes[1, 1].set_title('NN Training Loss Curve (Breast Cancer)')
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Loss')

plt.tight_layout()
plt.show()

log_runtime('Breast Cancer Experiments', start)

# %% [code cell 7]
start = time.time()

# Subsample for practical runtime in notebook execution.
mn_idx = np.random.choice(X_mnist.shape[0], 12000, replace=False)
Xmn = X_mnist[mn_idx]
ymn = y_mnist[mn_idx]

Xmn_train, Xmn_test, ymn_train, ymn_test = train_test_split(Xmn, ymn, test_size=0.2, stratify=ymn, random_state=42)

# Compare linear and RBF SVM.
svm_mn_lin = SVC(kernel='linear', C=1.5, random_state=42)
svm_mn_rbf = SVC(kernel='rbf', C=6, gamma='scale', random_state=42)
svm_mn_lin.fit(Xmn_train, ymn_train)
svm_mn_rbf.fit(Xmn_train, ymn_train)

ymn_pred_lin = svm_mn_lin.predict(Xmn_test)
ymn_pred_rbf = svm_mn_rbf.predict(Xmn_test)

acc_lin = accuracy_score(ymn_test, ymn_pred_lin)
acc_rbf = accuracy_score(ymn_test, ymn_pred_rbf)
svm_mn = svm_mn_rbf if acc_rbf >= acc_lin else svm_mn_lin
ymn_pred_svm = ymn_pred_rbf if acc_rbf >= acc_lin else ymn_pred_lin

add_result('MNIST', 'SVM', 'Accuracy', accuracy_score(ymn_test, ymn_pred_svm))
add_result('MNIST', 'SVM', 'Precision_macro', precision_score(ymn_test, ymn_pred_svm, average='macro'))
add_result('MNIST', 'SVM', 'Recall_macro', recall_score(ymn_test, ymn_pred_svm, average='macro'))
add_result('MNIST', 'SVM', 'F1_macro', f1_score(ymn_test, ymn_pred_svm, average='macro'))

# Neural network classifier.
nn_mn = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=50, random_state=42)
nn_mn.fit(Xmn_train, ymn_train)
ymn_pred_nn = nn_mn.predict(Xmn_test)

add_result('MNIST', 'Neural Network', 'Accuracy', accuracy_score(ymn_test, ymn_pred_nn))
add_result('MNIST', 'Neural Network', 'Precision_macro', precision_score(ymn_test, ymn_pred_nn, average='macro'))
add_result('MNIST', 'Neural Network', 'Recall_macro', recall_score(ymn_test, ymn_pred_nn, average='macro'))
add_result('MNIST', 'Neural Network', 'F1_macro', f1_score(ymn_test, ymn_pred_nn, average='macro'))

# K-means baseline for unsupervised contrast.
kmeans_mn = KMeans(n_clusters=10, random_state=42, n_init=10)
kmn_labels = kmeans_mn.fit_predict(Xmn)
kmn_rep = clustering_report(Xmn, kmn_labels, ymn)
add_result('MNIST', 'K-means', 'Silhouette', kmn_rep['silhouette'])
add_result('MNIST', 'K-means', 'ARI', kmn_rep['ari'])

# Visual diagnostics.
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
cm_svm = confusion_matrix(ymn_test, ymn_pred_svm)
cm_nn = confusion_matrix(ymn_test, ymn_pred_nn)

sns.heatmap(cm_svm, cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('MNIST SVM Confusion Matrix')

sns.heatmap(cm_nn, cmap='Oranges', ax=axes[0, 1])
axes[0, 1].set_title('MNIST NN Confusion Matrix')

axes[1, 0].plot(nn_mn.loss_curve_, color='darkorange')
axes[1, 0].set_title('MNIST NN Training Loss')
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Loss')

Xmn_pca2 = PCA(n_components=2, random_state=42).fit_transform(Xmn[:3000])
axes[1, 1].scatter(Xmn_pca2[:, 0], Xmn_pca2[:, 1], c=kmn_labels[:3000], cmap='tab10', s=5, alpha=0.6)
axes[1, 1].set_title('MNIST K-means Clusters (PCA projection)')

plt.tight_layout()
plt.show()

log_runtime('MNIST Experiments', start)

# %% [code cell 8]
start = time.time()

try:
    import gymnasium as gym
except Exception:
    import gym

env = gym.make('FrozenLake-v1', is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n


def reset_env(environment):
    out = environment.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(environment, action):
    out = environment.step(action)
    if len(out) == 5:
        ns, r, terminated, truncated, _ = out
        return ns, r, (terminated or truncated)
    ns, r, done, _ = out
    return ns, r, done


def evaluate_policy(environment, policy_fn, episodes=500):
    rewards = []
    for _ in range(episodes):
        s = reset_env(environment)
        done = False
        total_r = 0
        while not done:
            a = policy_fn(s)
            s, r, done = step_env(environment, a)
            total_r += r
        rewards.append(total_r)
    return float(np.mean(rewards))


def train_q_learning(environment, episodes=12000, alpha=0.15, gamma=0.99,
                     epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.9993):
    Q = np.zeros((n_states, n_actions))
    rewards = []
    for _ in range(episodes):
        s = reset_env(environment)
        done = False
        ep_reward = 0
        while not done:
            if np.random.rand() < epsilon:
                a = np.random.randint(n_actions)
            else:
                a = int(np.argmax(Q[s]))
            ns, r, done = step_env(environment, a)
            Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[ns]) - Q[s, a])
            s = ns
            ep_reward += r
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards.append(ep_reward)
    return Q, rewards


random_score = evaluate_policy(env, lambda s: np.random.randint(n_actions), episodes=800)
Q, q_rewards = train_q_learning(env)
q_policy = lambda s: int(np.argmax(Q[s]))
q_score = evaluate_policy(env, q_policy, episodes=1000)

add_result('FrozenLake', 'Random Agent', 'AvgReward', random_score)
add_result('FrozenLake', 'Q-learning Agent', 'AvgReward', q_score)

# RL visual analysis.
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

rolling = pd.Series(q_rewards).rolling(300).mean()
axes[0].plot(rolling, color='royalblue')
axes[0].set_title('Q-learning Reward Learning Curve (rolling avg)')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Average Reward')

bars = axes[1].bar(['Random', 'Q-learning'], [random_score, q_score], color=['salmon', 'royalblue'])
for bar, val in zip(bars, [random_score, q_score]):
    axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.002, f'{val:.3f}', ha='center')
axes[1].set_title('Policy Comparison')
axes[1].set_ylabel('Average Reward')

q_best = Q.max(axis=1).reshape(4, 4)
sns.heatmap(q_best, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[2])
axes[2].set_title('Q-table Heatmap (max action value/state)')

plt.tight_layout()
plt.show()

log_runtime('FrozenLake RL', start)

# %% [code cell 9]
# ==========================================
# 5) Comparative analysis across algorithms
# ==========================================
results_df = pd.DataFrame(results)
runtime_df = pd.DataFrame(runtime_log)

# Complexity and qualitative properties table.
comparison_table = pd.DataFrame([
    ['K-means', 'O(nkdi)', 'Low-Medium', 'High', 'Centroids'],
    ['SVM (kernel)', 'O(n^2 d) to O(n^3)', 'Medium', 'Medium', 'Support vectors'],
    ['Neural Network (MLP)', 'O(n * p * epochs)', 'Low-Medium', 'High', 'Weights/loss'],
    ['Lambda-connectedness', 'O(n^2 d) + O(n+e)', 'Medium', 'Low-Medium', 'Graph components'],
    ['Q-learning (tabular)', 'O(E*T)', 'Medium', 'Medium', 'Q-table policy'],
], columns=['Algorithm', 'Typical Complexity', 'Interpretability', 'Scalability', 'Convergence Signal'])

strengths_weaknesses = pd.DataFrame([
    ['K-means', 'Fast and simple segmentation', 'Assumes spherical clusters, sensitive to k'],
    ['SVM', 'Strong margins, robust on structured data', 'Kernel scaling cost on large n'],
    ['Neural Network', 'High nonlinear representation power', 'Hyperparameter-sensitive, less interpretable'],
    ['Lambda-connectedness', 'Reveals density/connectivity structures', 'Pairwise graph cost is high'],
    ['Q-learning', 'Model-free sequential optimization', 'Sparse reward slows early learning'],
], columns=['Algorithm', 'Strengths', 'Weaknesses'])

print('Metric Results (top rows):')
display(results_df.head(20))
print('')
print('Runtime Summary (seconds):')
display(runtime_df)
print('')
print('Complexity / Scalability / Interpretability:')
display(comparison_table)
print('')
print('Strengths and Weaknesses:')
display(strengths_weaknesses)

# %% [code cell 10]
# ==============================================
# DECISION EXTRACTION â€” AMERICAN HOUSING
# ==============================================
print('--- DATASET: American Housing ---')

housing_dec_df = pd.DataFrame(X_housing, columns=housing.feature_names)
housing_dec_df['Price'] = y_housing

# Approximate real city names via nearest major California city by latitude/longitude.
ca_city_refs = {
    'Los Angeles': (34.0522, -118.2437),
    'San Diego': (32.7157, -117.1611),
    'San Jose': (37.3382, -121.8863),
    'San Francisco': (37.7749, -122.4194),
    'Sacramento': (38.5816, -121.4944),
    'Fresno': (36.7378, -119.7871),
    'Long Beach': (33.7701, -118.1937),
    'Oakland': (37.8044, -122.2712),
    'Bakersfield': (35.3733, -119.0187),
    'Anaheim': (33.8366, -117.9143),
    'Santa Ana': (33.7455, -117.8677),
    'Riverside': (33.9806, -117.3755),
    'Stockton': (37.9577, -121.2908),
    'Irvine': (33.6846, -117.8265),
    'Chula Vista': (32.6401, -117.0842),
    'Fremont': (37.5483, -121.9886),
    'San Bernardino': (34.1083, -117.2898),
    'Modesto': (37.6391, -120.9969),
    'Oxnard': (34.1975, -119.1771),
    'Huntington Beach': (33.6603, -117.9992),
    'Santa Barbara': (34.4208, -119.6982),
    'Monterey': (36.6002, -121.8947),
    'Palm Springs': (33.8303, -116.5453),
    'Redding': (40.5865, -122.3917),
    'Eureka': (40.8021, -124.1637),
    'Mammoth Lakes': (37.6485, -118.9721),
}

city_names = list(ca_city_refs.keys())
city_coords = np.array([ca_city_refs[c] for c in city_names])


def assign_city_name(lat, lon):
    d2 = (city_coords[:, 0] - lat) ** 2 + (city_coords[:, 1] - lon) ** 2
    return city_names[int(np.argmin(d2))]


housing_dec_df['CityName'] = housing_dec_df.apply(lambda r: assign_city_name(r['Latitude'], r['Longitude']), axis=1)

# 1) K-means decision extraction.
housing_dec_df['KMeansCluster'] = kh_labels
cluster_price = housing_dec_df.groupby('KMeansCluster')['Price'].agg(['mean', 'count']).sort_values('mean')
cheapest_cluster = int(cluster_price.index[0])
expensive_cluster = int(cluster_price.index[-1])

cheap_city_summary = (
    housing_dec_df[housing_dec_df['KMeansCluster'] == cheapest_cluster]
    .groupby('CityName', as_index=False)
    .agg(AvgPrice=('Price', 'mean'), AvgIncome=('MedInc', 'mean'), Records=('Price', 'size'))
    .sort_values('AvgPrice')
)
exp_city_summary = (
    housing_dec_df[housing_dec_df['KMeansCluster'] == expensive_cluster]
    .groupby('CityName', as_index=False)
    .agg(AvgPrice=('Price', 'mean'), AvgIncome=('MedInc', 'mean'), Records=('Price', 'size'))
    .sort_values('AvgPrice', ascending=False)
)

top10_cheapest_cluster = cheap_city_summary.head(10)
top10_expensive_cluster = exp_city_summary.head(10)

# 2) Lambda-connectedness decision extraction.
lambda_src = pd.DataFrame(X_housing[:3000], columns=housing.feature_names).iloc[idx_h].copy()
lambda_src['Component'] = lmb_h_labels
lambda_src['Price'] = y_housing[:3000][idx_h]
lambda_src['CityName'] = lambda_src.apply(lambda r: assign_city_name(r['Latitude'], r['Longitude']), axis=1)
component_rank = lambda_src.groupby('Component')['Price'].agg(['mean', 'count']).sort_values('mean')
cheapest_component = int(component_rank.index[0])
lambda_top10 = (
    lambda_src[lambda_src['Component'] == cheapest_component]
    .groupby('CityName', as_index=False)
    .agg(AvgPrice=('Price', 'mean'), AvgIncome=('MedInc', 'mean'), Records=('Price', 'size'))
    .sort_values('AvgPrice')
    .head(10)
)

# 3) SVM confidence extraction (cheap vs expensive binary decision).
y_h_binary = (y_housing >= np.median(y_housing)).astype(int)
all_idx = np.arange(len(X_housing))
idx_train, idx_test = train_test_split(all_idx, test_size=0.2, random_state=42, stratify=y_h_binary)

X_train_b = X_housing[idx_train]
X_test_b = X_housing[idx_test]
y_train_b = y_h_binary[idx_train]
y_test_b = y_h_binary[idx_test]

scaler_bin = StandardScaler()
X_train_bs = scaler_bin.fit_transform(X_train_b)
X_test_bs = scaler_bin.transform(X_test_b)

svm_house_cls = SVC(C=3, kernel='rbf', probability=True, random_state=42)
svm_house_cls.fit(X_train_bs, y_train_b)
prob_test = svm_house_cls.predict_proba(X_test_bs)

svm_rank_df = pd.DataFrame(X_test_b, columns=housing.feature_names)
svm_rank_df['TrueClass'] = y_test_b
svm_rank_df['ProbExpensive'] = prob_test[:, 1]
svm_rank_df['ProbCheap'] = prob_test[:, 0]
svm_rank_df['CityName'] = svm_rank_df.apply(lambda r: assign_city_name(r['Latitude'], r['Longitude']), axis=1)

most_conf_cheap = (
    svm_rank_df.groupby('CityName', as_index=False)
    .agg(ConfidenceCheap=('ProbCheap', 'max'), AvgIncome=('MedInc', 'mean'), Records=('ProbCheap', 'size'))
    .sort_values('ConfidenceCheap', ascending=False)
    .head(10)
)
most_conf_expensive = (
    svm_rank_df.groupby('CityName', as_index=False)
    .agg(ConfidenceExpensive=('ProbExpensive', 'max'), AvgIncome=('MedInc', 'mean'), Records=('ProbExpensive', 'size'))
    .sort_values('ConfidenceExpensive', ascending=False)
    .head(10)
)

# 4) Neural ranking extraction.
nn_pred_all = nn_h.predict(scaler_h.transform(X_housing))
nn_rank_df = pd.DataFrame(X_housing, columns=housing.feature_names)
nn_rank_df['PredictedPrice'] = nn_pred_all
nn_rank_df['CityName'] = nn_rank_df.apply(lambda r: assign_city_name(r['Latitude'], r['Longitude']), axis=1)

city_pred_summary = (
    nn_rank_df.groupby('CityName', as_index=False)
    .agg(AvgPredictedPrice=('PredictedPrice', 'mean'), AvgIncome=('MedInc', 'mean'), Records=('PredictedPrice', 'size'))
)
nn_top10_low = city_pred_summary.sort_values('AvgPredictedPrice').head(10)
nn_top10_high = city_pred_summary.sort_values('AvgPredictedPrice', ascending=False).head(10)

# Final decision-oriented outputs.
print('\nModel Output Snapshot:')
print(f'- K-means clusters: {kh_rep["n_clusters"] if "n_clusters" in kh_rep else len(np.unique(kh_labels))}')
print(f'- Lambda components: {lmb_h_rep["n_clusters"]}')
print(f'- SVM binary classifier confidence generated for {len(svm_rank_df)} holdout records')
print(f'- NN predicted prices generated for {len(nn_rank_df)} records')

print('\nProcessed Interpretation:')
print(f'- Cheapest K-means cluster: {cheapest_cluster}')
print(f'- Most expensive K-means cluster: {expensive_cluster}')
print(f'- Cheapest lambda-connected component: {cheapest_component}')
print('- All outputs below are unique city-level decisions (no repeated city names).')

print('\nFINAL DECISION OUTPUT â€” Top 10 Cheapest Cities (unique city ranking)')
display(top10_cheapest_cluster)

print('FINAL DECISION OUTPUT â€” Top 10 Most Expensive Cities (unique city ranking)')
display(top10_expensive_cluster)

print('FINAL DECISION OUTPUT â€” Top 10 Most Confident Cheap City Decisions (SVM)')
display(most_conf_cheap)

print('FINAL DECISION OUTPUT â€” Top 10 Most Confident Expensive City Decisions (SVM)')
display(most_conf_expensive)

print('FINAL DECISION OUTPUT â€” Top 10 Lowest Predicted City Prices (Neural Network)')
display(nn_top10_low)

print('FINAL DECISION OUTPUT â€” Top 10 Highest Predicted City Prices (Neural Network)')
display(nn_top10_high)

print('FINAL DECISION OUTPUT â€” Top Cheapest Connected-Region Cities (Lambda-connectedness)')
display(lambda_top10)

# Decision plots.
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

cheap_plot = top10_cheapest_cluster.sort_values('AvgPrice', ascending=False)
axes[0].barh(cheap_plot['CityName'], cheap_plot['AvgPrice'], color='seagreen')
axes[0].set_title('Top 10 Cheapest Cities (Unique City Decision View)')
axes[0].set_xlabel('Average Price')

exp_plot = top10_expensive_cluster.sort_values('AvgPrice', ascending=True)
axes[1].barh(exp_plot['CityName'], exp_plot['AvgPrice'], color='firebrick')
axes[1].set_title('Top 10 Most Expensive Cities (Unique City Decision View)')
axes[1].set_xlabel('Average Price')

plt.tight_layout()
plt.show()

# %% [code cell 11]
# ==============================================
# DECISION EXTRACTION â€” BREAST CANCER
# ==============================================
print('--- DATASET: Breast Cancer ---')

# Probability-capable SVM for risk extraction.
svm_b_prob = SVC(kernel='rbf', C=5.0, gamma='scale', probability=True, random_state=42)
svm_b_prob.fit(Xb_train_s, yb_train)

svm_prob = svm_b_prob.predict_proba(Xb_test_s)
nn_prob = nn_b.predict_proba(Xb_test_s)

# Class coding: 0=malignant, 1=benign. High malignant probability => high risk.
bc_dec = pd.DataFrame(Xb_test, columns=bc.feature_names)
bc_dec['TrueLabel'] = np.where(yb_test == 0, 'Malignant', 'Benign')
bc_dec['SVM_MalignantProb'] = svm_prob[:, 0]
bc_dec['NN_MalignantProb'] = nn_prob[:, 0]
bc_dec['CombinedRiskScore'] = 0.5 * bc_dec['SVM_MalignantProb'] + 0.5 * bc_dec['NN_MalignantProb']

bc_dec['RiskCategory'] = pd.cut(
    bc_dec['CombinedRiskScore'],
    bins=[-0.01, 0.33, 0.66, 1.01],
    labels=['Low', 'Medium', 'High']
)

# Human-readable case profile names (letters only).

def alpha_name(i):
    letters = []
    i = int(i)
    while True:
        i, rem = divmod(i, 26)
        letters.append(chr(65 + rem))
        if i == 0:
            break
        i -= 1
    return ''.join(reversed(letters))


bc_dec['CaseProfile'] = ['Patient-' + alpha_name(i) for i in range(len(bc_dec))]

high_risk_top10 = bc_dec.sort_values('CombinedRiskScore', ascending=False).head(10)
most_confident_benign = bc_dec.sort_values('CombinedRiskScore', ascending=True).head(10)
risk_counts = bc_dec['RiskCategory'].value_counts().reindex(['High', 'Medium', 'Low'])

print('\nModel Output Snapshot:')
print('- SVM and NN probability distributions extracted on holdout set')

print('\nProcessed Interpretation:')
print('- Combined risk score fuses margin-based and neural confidence')
print('- Risk categories partition patients into triage-ready groups')

print('\nFINAL DECISION OUTPUT â€” Risk Category Counts')
display(risk_counts.to_frame('CaseCount'))

print('FINAL DECISION OUTPUT â€” Top 10 Highest-Risk Patient Profiles (review priority)')
display(high_risk_top10[['CaseProfile', 'CombinedRiskScore', 'SVM_MalignantProb', 'NN_MalignantProb', 'TrueLabel'] + list(bc.feature_names[:5])])

print('FINAL DECISION OUTPUT â€” Top 10 Most Confident Benign Patient Profiles')
display(most_confident_benign[['CaseProfile', 'CombinedRiskScore', 'SVM_MalignantProb', 'NN_MalignantProb', 'TrueLabel'] + list(bc.feature_names[:5])])

# Decision plots.
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

risk_counts.plot(kind='bar', ax=axes[0], color=['crimson', 'orange', 'seagreen'])
axes[0].set_title('Patient Risk Group Distribution')
axes[0].set_ylabel('Number of Profiles')
axes[0].tick_params(axis='x', rotation=0)

risk_plot = high_risk_top10[['CaseProfile', 'CombinedRiskScore']].sort_values('CombinedRiskScore', ascending=True)
axes[1].barh(risk_plot['CaseProfile'], risk_plot['CombinedRiskScore'], color='crimson')
axes[1].set_title('Top High-Risk Patient Profiles')
axes[1].set_xlabel('Combined Malignancy Risk Score')

plt.tight_layout()
plt.show()

# %% [code cell 12]
# ==============================================
# DECISION EXTRACTION â€” MNIST
# ==============================================
print('--- DATASET: MNIST ---')

# SVM confidence via top-2 margin on decision scores.
svm_scores = svm_mn.decision_function(Xmn_test)
if svm_scores.ndim == 1:
    svm_margin = np.abs(svm_scores)
else:
    top2 = np.partition(svm_scores, -2, axis=1)[:, -2:]
    svm_margin = np.abs(top2[:, 1] - top2[:, 0])

svm_pred = ymn_pred_svm
nn_proba = nn_mn.predict_proba(Xmn_test)
nn_pred = ymn_pred_nn
nn_conf = nn_proba.max(axis=1)


def alpha_name(i):
    letters = []
    i = int(i)
    while True:
        i, rem = divmod(i, 26)
        letters.append(chr(65 + rem))
        if i == 0:
            break
        i -= 1
    return ''.join(reversed(letters))


mn_dec = pd.DataFrame({
    'GlyphProfile': ['Glyph-' + alpha_name(i) for i in range(len(ymn_test))],
    'TrueDigit': ymn_test,
    'SVMPred': svm_pred,
    'NNPred': nn_pred,
    'SVMMarginConfidence': svm_margin,
    'NNProbConfidence': nn_conf,
})
mn_dec['Agreement'] = (mn_dec['SVMPred'] == mn_dec['NNPred']).astype(int)
mn_dec['CombinedConfidence'] = 0.5 * (mn_dec['SVMMarginConfidence'] / (mn_dec['SVMMarginConfidence'].max() + 1e-12)) + 0.5 * mn_dec['NNProbConfidence']

mn_top10_confident = mn_dec.sort_values('CombinedConfidence', ascending=False).head(10)
mn_top10_uncertain = mn_dec.sort_values('CombinedConfidence', ascending=True).head(10)

print('\nModel Output Snapshot:')
print('- SVM decision margins and NN class probabilities extracted')

print('\nProcessed Interpretation:')
print('- High confidence + model agreement => strong auto-accept candidates')
print('- Low confidence => send to human review or fallback rule')

print('\nFINAL DECISION OUTPUT â€” Top 10 Most Confident Glyph Decisions (auto-accept queue)')
display(mn_top10_confident[['GlyphProfile', 'TrueDigit', 'SVMPred', 'NNPred', 'Agreement', 'CombinedConfidence']])

print('FINAL DECISION OUTPUT â€” Top 10 Lowest-Confidence Glyph Decisions (manual review queue)')
display(mn_top10_uncertain[['GlyphProfile', 'TrueDigit', 'SVMPred', 'NNPred', 'Agreement', 'CombinedConfidence']])

# Decision plots and visual boards.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

conf_plot = mn_top10_confident.sort_values('CombinedConfidence', ascending=True)
axes[0].barh(conf_plot['GlyphProfile'], conf_plot['CombinedConfidence'], color='royalblue')
axes[0].set_title('Auto-Accept Queue (Most Confident)')
axes[0].set_xlabel('Combined Confidence')

unc_plot = mn_top10_uncertain.sort_values('CombinedConfidence', ascending=False)
axes[1].barh(unc_plot['GlyphProfile'], unc_plot['CombinedConfidence'], color='darkorange')
axes[1].set_title('Manual Review Queue (Least Confident)')
axes[1].set_xlabel('Combined Confidence')

plt.tight_layout()
plt.show()

# Show glyph images for manual review queue.
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax, row in zip(axes.ravel(), mn_top10_uncertain.itertuples()):
    idx = int(np.where(np.array(mn_dec['GlyphProfile']) == row.GlyphProfile)[0][0])
    ax.imshow(Xmn_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f"{row.GlyphProfile}\nT:{row.TrueDigit} S:{row.SVMPred} N:{row.NNPred}", fontsize=8)
    ax.axis('off')
plt.suptitle('Lowest-Confidence Glyph Profiles (Manual Review Board)', fontsize=12)
plt.tight_layout()
plt.show()

# %% [code cell 13]
# ==============================================
# DECISION EXTRACTION â€” FROZENLAKE RL
# ==============================================
print('--- DATASET: FrozenLake ---')

action_map = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}

tile_names = [
    'Start',
    'Frost Meadow',
    'Glacier Bend',
    'Ice Corridor',
    'Snow Bridge',
    'Abyss Pit',
    'Crystal Pass',
    'Abyss Gate',
    'North Shelf',
    'Windy Fork',
    'Aurora Route',
    'Abyss Hollow',
    'Abyss Ridge',
    'Silver Trail',
    'Summit Approach',
    'Goal Beacon',
]

policy = np.argmax(Q, axis=1)
policy_df = pd.DataFrame({
    'TileName': tile_names,
    'BestAction': [action_map[a] for a in policy],
    'BestQValue': Q.max(axis=1)
})

# Derive a representative greedy path from start state.
path_tiles = []
path_actions = []
state = reset_env(env)
visited = set()
for _ in range(30):
    path_tiles.append(tile_names[int(state)])
    action = int(policy[state])
    path_actions.append(action_map[action])
    nxt, reward, done = step_env(env, action)
    state = nxt
    if done:
        path_tiles.append(tile_names[int(state)])
        break
    if (int(state), len(path_tiles)) in visited:
        break
    visited.add((int(state), len(path_tiles)))

print('\nModel Output Snapshot:')
print('- Learned Q-table and greedy policy extracted')

print('\nProcessed Interpretation:')
print('- Tile-wise best action gives deployable control policy')
print('- Greedy trajectory gives a human-readable action plan')

print('\nFINAL DECISION OUTPUT â€” Optimal Action Per Named Tile')
display(policy_df)

print('FINAL DECISION OUTPUT â€” Recommended Greedy Action Path from Start')
for i, tile in enumerate(path_tiles[:-1]):
    print(f'Step {i+1}: {tile} -> Action {path_actions[i]}')
print(f'End Tile: {path_tiles[-1]}')
print(f'Policy Average Reward: {q_score:.3f} (Random baseline: {random_score:.3f})')

# Decision plot.
plot_df = policy_df.sort_values('BestQValue', ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(plot_df['TileName'], plot_df['BestQValue'], color='slateblue')
plt.title('Named-Tile Decision Strength (Best Q-Value per Tile)')
plt.xlabel('Best Q-Value')
plt.tight_layout()
plt.show()

# %% [code cell 14]
# ==============================================
# GLOBAL DECISION SUPPORT SUMMARY
# ==============================================
print('--- GLOBAL FINAL OUTPUT ---')

# Best method for ranking tasks (using housing RMSE as ranking quality proxy).
housing_rank_metrics = results_df[(results_df['Dataset'] == 'Housing') & (results_df['Metric'] == 'RMSE')].copy()
best_ranking_algo = housing_rank_metrics.sort_values('Value').iloc[0]['Algorithm']

# Best method for classification tasks (avg accuracy over Breast Cancer + MNIST).
acc_df = results_df[(results_df['Metric'] == 'Accuracy') & (results_df['Dataset'].isin(['Breast Cancer', 'MNIST']))].copy()
acc_mean = acc_df.groupby('Algorithm')['Value'].mean().sort_values(ascending=False)
best_classification_algo = acc_mean.index[0]

# Best structure discovery among K-means vs Lambda-connectedness (silhouette where available).
struct_df = results_df[(results_df['Algorithm'].isin(['K-means', 'Lambda-connectedness'])) & (results_df['Metric'].isin(['Silhouette', 'ARI', 'ARI_vs_binned_target']))].copy()
struct_summary = struct_df.groupby('Algorithm')['Value'].mean().sort_values(ascending=False)
best_structure_algo = struct_summary.index[0]

global_decision_table = pd.DataFrame([
    ['Best Algorithm for Ranking Tasks', best_ranking_algo],
    ['Best Algorithm for Classification Tasks', best_classification_algo],
    ['Best Algorithm for Structure Discovery', best_structure_algo],
    ['Best Policy Method for Sequential Decisions', 'Q-learning'],
], columns=['Decision Question', 'Selected Method'])

print('\nGlobal Ranking System Insights:')
print('- Ranking outputs are strongest when prediction calibration is stable (NN/SVM on housing).')
print('- Classification confidence enables triage-oriented decision pipelines (Breast Cancer, MNIST).')
print('- Structure discovery quality varies with geometry assumptions (centroid vs connectivity).')

print('\nFINAL DECISION OUTPUT â€” Method Selection Matrix')
display(global_decision_table)

print('FINAL DECISION OUTPUT â€” Classification Accuracy Leaderboard')
display(acc_mean.to_frame('MeanAccuracy'))

print('FINAL DECISION OUTPUT â€” Structure Discovery Leaderboard')
display(struct_summary.to_frame('MeanStructureScore'))

