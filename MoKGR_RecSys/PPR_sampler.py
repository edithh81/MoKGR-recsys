from tqdm import tqdm
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse

class PPRSampler:
    def __init__(self, n_ent, edges, sampling_percentage=0.8, PPR_alpha=0.85, max_iter=100, tol=1e-9):
        print('==> Initializing PPRSampler...')
        self.n_ent = n_ent
        self.edges = edges
        self.sampling_percentage = sampling_percentage
        self.PPR_alpha = PPR_alpha
        self.max_iter = max_iter
        self.tol = tol

        self.adjacency_matrix = self._build_adjacency_matrix()
        print('==> Initialization completed.')

    def _build_adjacency_matrix(self):
        src, dst = zip(*self.edges)
        src = cp.array(src, dtype=cp.int32)
        dst = cp.array(dst, dtype=cp.int32)
        data = cp.ones(len(src), dtype=cp.float32)

        adjacency_matrix = cpx_sparse.coo_matrix((data, (src, dst)), shape=(self.n_ent, self.n_ent))

        # Undirected graph
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T

        # Convert to CSR for efficient arithmetic
        adjacency_matrix = adjacency_matrix.tocsr()

        # Row-normalize
        row_sums = cp.array(adjacency_matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        # Build a diagonal matrix of inverse row sums and left-multiply
        inv_row_sums = cpx_sparse.diags(1.0 / row_sums)
        adjacency_matrix = inv_row_sums @ adjacency_matrix

        return adjacency_matrix

    def compute_ppr_for_seed(self, seed):
        teleport = cp.zeros(self.n_ent, dtype=cp.float32)
        teleport[seed] = 1.0

        pr = cp.ones(self.n_ent, dtype=cp.float32) / self.n_ent

        for i in range(self.max_iter):
            new_pr = self.PPR_alpha * self.adjacency_matrix.dot(pr) + (1 - self.PPR_alpha) * teleport
            if cp.linalg.norm(new_pr - pr, ord=1) < self.tol:
                pr = new_pr
                break
            pr = new_pr

        return cp.asnumpy(pr)

    def sample_nodes(self, seeds):
        ppr_scores_per_seed = {}
        print('==> Calculating PPR Scores Using GPUs...')
        for seed in tqdm(seeds, desc='Calculating PPR'):
            ppr_scores_per_seed[seed] = self.compute_ppr_for_seed(seed)
        return ppr_scores_per_seed