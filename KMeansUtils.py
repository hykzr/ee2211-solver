import numpy as np

from CommonUtils import ensure_2d, pretty_print_array


def validate_kmeans_inputs(X, centroids, max_iterations=None):
	X = ensure_2d(X)
	centroids = ensure_2d(centroids)

	if X.size == 0:
		raise ValueError("Input matrix must contain at least one point.")
	if centroids.size == 0:
		raise ValueError("Centroid matrix must contain at least one centroid.")
	if X.shape[1] != centroids.shape[1]:
		raise ValueError(
			f"Feature dimension mismatch: input points have {X.shape[1]} column(s) "
			f"but centroids have {centroids.shape[1]} column(s)."
		)
	if not np.all(np.isfinite(X)):
		raise ValueError("Input matrix must contain only finite numeric values.")
	if not np.all(np.isfinite(centroids)):
		raise ValueError("Centroid matrix must contain only finite numeric values.")
	if max_iterations is not None:
		if int(max_iterations) != max_iterations or max_iterations < 1:
			raise ValueError("Max iterations must be a positive integer or left empty.")

	return X, centroids, None if max_iterations is None else int(max_iterations)


def k_means(X, initial_centroids, max_iterations=None):
	X, centroids, max_iterations = validate_kmeans_inputs(X, initial_centroids, max_iterations)
	centroids = centroids.copy()
	history = []
	iteration = 0

	while max_iterations is None or iteration < max_iterations:
		iteration += 1
		distances = euclidean_distance_matrix(X, centroids)
		assignments = np.argmin(distances, axis=1)
		next_centroids = update_centroids(X, assignments, centroids)
		converged = np.allclose(next_centroids, centroids, atol=1e-10, rtol=1e-10)

		history.append(
			{
				"iteration": iteration,
				"start_centroids": centroids.copy(),
				"distances": distances,
				"assignments": assignments.copy(),
				"centroids": next_centroids.copy(),
				"converged": converged,
			}
		)

		centroids = next_centroids
		if converged:
			break

	return history


def euclidean_distance_matrix(X, centroids):
	diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
	return np.linalg.norm(diff, axis=2)


def update_centroids(X, assignments, previous_centroids):
	next_centroids = previous_centroids.copy()
	for cluster_index in range(previous_centroids.shape[0]):
		members = X[assignments == cluster_index]
		if members.size:
			next_centroids[cluster_index] = np.mean(members, axis=0)
	return next_centroids


def print_k_means_result(history):
	for item in history:
		print(f"\nIteration {item['iteration']}")
		pretty_print_array("centroids", item["centroids"], show_python=False)
		pretty_print_array("cluster", (item["assignments"] + 1).reshape(-1, 1), show_python=False)
		if item["converged"]:
			print("converged")
