#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// A struct to hold a token's index and its probability.
typedef struct {
	int	  index;
	float prob;
} ProbIndex;

// Comparison function for qsort to sort tokens by probability in descending order.
int compare_prob(const void *a, const void *b)
{
	ProbIndex *pi1 = (ProbIndex *)a;
	ProbIndex *pi2 = (ProbIndex *)b;
	if (pi1->prob > pi2->prob)
		return -1;
	if (pi1->prob < pi2->prob)
		return 1;
	return 0;
}

// Computes the softmax of an array of logits, returning a new array of probabilities.
static float *softmax(float *logits, int size)
{
	float *probs = malloc(size * sizeof(float));
	if (!probs)
		return NULL;

	float max_logit = -INFINITY;
	for (int i = 0; i < size; i++) {
		if (logits[i] > 80.0f)
			logits[i] = 80.0f;
		if (logits[i] < -80.0f)
			logits[i] = -80.0f;

		if (logits[i] > max_logit) {
			max_logit = logits[i];
		}
	}

	// Handle case where all logits are -infinity
	if (max_logit == -INFINITY) {
		for (int i = 0; i < size; i++) {
			probs[i] = 1.0f / size;
		}
		return probs;
	}

	float sum = 0.0f;
	for (int i = 0; i < size; i++) {
		// Subtracting max_logit before exponentiating is a trick for numerical stability
		probs[i] = expf(logits[i] - max_logit);
		sum += probs[i];
	}
	for (int i = 0; i < size; i++) {
		probs[i] /= sum;
	}

	return probs;
}

// Samples a token index from a probability distribution.
int sample_from_probs(float *probs, int size)
{
	float r	  = (float)rand() / RAND_MAX;
	float cdf = 0.0f;
	for (int i = 0; i < size; i++) {
		cdf += probs[i];
		if (cdf >= r) {
			return i;
		}
	}
	return size - 1; // Fallback in case of floating point inaccuracies
}

// Performs top-k sampling.
int top_k_sampling(float *probs, int size, int k)
{
	if (k <= 0 || k > size) {
		k = size;
	}

	ProbIndex *pi = malloc(size * sizeof(ProbIndex));
	if (!pi)
		return sample_from_probs(probs, size); // Fallback

	for (int i = 0; i < size; i++) {
		pi[i].index = i;
		pi[i].prob	= probs[i];
	}
	qsort(pi, size, sizeof(ProbIndex), compare_prob);

	// Renormalize the probabilities of the top k tokens
	float sum = 0.0f;
	for (int i = 0; i < k; i++) {
		sum += pi[i].prob;
	}
	for (int i = 0; i < k; i++) {
		pi[i].prob /= sum;
	}

	// Sample from the truncated and renormalized distribution
	float r		= (float)rand() / RAND_MAX;
	float cdf	= 0.0f;
	int	  index = pi[k - 1].index; // Default to the last token in the top-k set
	for (int i = 0; i < k; i++) {
		cdf += pi[i].prob;
		if (cdf >= r) {
			index = pi[i].index;
			break;
		}
	}
	free(pi);
	return index;
}

// Performs nucleus (top-p) sampling.
int nucleus_sampling(float *probs, int size, float p)
{
	ProbIndex *pi = malloc(size * sizeof(ProbIndex));
	if (!pi)
		return sample_from_probs(probs, size); // Fallback

	for (int i = 0; i < size; i++) {
		pi[i].index = i;
		pi[i].prob	= probs[i];
	}
	qsort(pi, size, sizeof(ProbIndex), compare_prob);

	// Find the cutoff point for the nucleus
	float cum_prob = 0.0f;
	int	  cutoff   = 0;
	for (int i = 0; i < size; i++) {
		cum_prob += pi[i].prob;
		if (cum_prob >= p) {
			cutoff = i + 1;
			break;
		}
	}
	if (cutoff == 0)
		cutoff = size; // Should not happen if probs sum to 1

	// Renormalize the probabilities within the nucleus
	float sum = 0.0f;
	for (int i = 0; i < cutoff; i++) {
		sum += pi[i].prob;
	}
	for (int i = 0; i < cutoff; i++) {
		pi[i].prob /= sum;
	}

	// Sample from the nucleus
	float r		= (float)rand() / RAND_MAX;
	float cdf	= 0.0f;
	int	  index = pi[cutoff - 1].index; // Default to the last token in the nucleus
	for (int i = 0; i < cutoff; i++) {
		cdf += pi[i].prob;
		if (cdf >= r) {
			index = pi[i].index;
			break;
		}
	}
	free(pi);
	return index;
}

// Returns the index of the maximum value in an array (greedy sampling).
int argmax(float *array, int size)
{
	int	  max_idx = 0;
	float max_val = array[0];
	for (int i = 1; i < size; i++) {
		if (array[i] > max_val) {
			max_val = array[i];
			max_idx = i;
		}
	}
	return max_idx;
}

int predict_next_token(float *logits, int vocab_size, const char *method, float temperature, int k, float p,
					   const int *prompt_tokens, int prompt_len, const int *generated_tokens, int gen_len,
					   float repetition_penalty)
{
	// 1. Apply repetition penalty to the logits
	if (repetition_penalty != 1.0f) {
		// Using a simple set for lookup, could be optimized further if needed
		for (int i = 0; i < prompt_len; i++) {
			int token_idx = prompt_tokens[i];
			if (token_idx >= 0 && token_idx < vocab_size) {
				if (logits[token_idx] > 0)
					logits[token_idx] /= repetition_penalty;
				else
					logits[token_idx] *= repetition_penalty;
			}
		}
		for (int i = 0; i < gen_len; i++) {
			int token_idx = generated_tokens[i];
			if (token_idx >= 0 && token_idx < vocab_size) {
				if (logits[token_idx] > 0)
					logits[token_idx] /= repetition_penalty;
				else
					logits[token_idx] *= repetition_penalty;
			}
		}
	}

	// 2. Apply temperature scaling to the logits
	if (temperature <= 0.0f) {
		// A temperature of 0 corresponds to greedy sampling (argmax)
		return argmax(logits, vocab_size);
	}
	for (int i = 0; i < vocab_size; i++) {
		logits[i] /= temperature;
	}

	// 3. Compute the final probability distribution via softmax
	float *probs = softmax(logits, vocab_size);
	if (!probs)
		return vocab_size - 1; // Fallback on error

	// 4. Select the sampling method based on the final probabilities
	int token;
	if (strcmp(method, "top_k") == 0) {
		token = top_k_sampling(probs, vocab_size, k);
	} else if (strcmp(method, "nucleus") == 0 || strcmp(method, "top_p") == 0) {
		token = nucleus_sampling(probs, vocab_size, p);
	} else {
		// Default to standard multinomial sampling if method is "temperature" or unrecognized
		token = sample_from_probs(probs, vocab_size);
	}

	free(probs);
	return token;
}

int sample_top_k_with_penalty(const float *logits_in, int vocab_size, const int *prev_tokens, int num_prev_tokens,
							  int top_k, float temperature, float repetition_penalty, unsigned int seed)
{
	if (top_k <= 0 || top_k > vocab_size)
		top_k = vocab_size;

	// Step 1: Copy and apply repetition penalty
	float *logits = (float *)malloc(sizeof(float) * vocab_size);
	memcpy(logits, logits_in, sizeof(float) * vocab_size);

	for (int i = 0; i < num_prev_tokens; ++i) {
		int token = prev_tokens[i];
		if (token >= 0 && token < vocab_size) {
			if (logits[token] < 0)
				logits[token] *= repetition_penalty;
			else
				logits[token] /= repetition_penalty;
		}
	}

	// Step 2: Scale by temperature
	for (int i = 0; i < vocab_size; ++i) {
		logits[i] /= temperature;
	}

	// Step 3: Partial sort to get top-k indices
	int *indices = (int *)malloc(sizeof(int) * vocab_size);
	for (int i = 0; i < vocab_size; ++i)
		indices[i] = i;

	// Selection sort for top-k
	for (int i = 0; i < top_k; ++i) {
		for (int j = i + 1; j < vocab_size; ++j) {
			if (logits[j] > logits[i]) {
				// Swap logits
				float tmp_val = logits[i];
				logits[i]	  = logits[j];
				logits[j]	  = tmp_val;

				// Swap indices
				int tmp_idx = indices[i];
				indices[i]	= indices[j];
				indices[j]	= tmp_idx;
			}
		}
	}

	// Step 4: Softmax over top-k
	float  max_logit = logits[0];
	float  sum_exp	 = 0.0f;
	float *probs	 = (float *)malloc(sizeof(float) * top_k);

	for (int i = 0; i < top_k; ++i) {
		float x	 = logits[i] - max_logit;
		probs[i] = expf(x);
		sum_exp += probs[i];
	}

	for (int i = 0; i < top_k; ++i) {
		probs[i] /= sum_exp;
	}

	// Step 5: Sample from top-k
	float r	  = (float)rand_r(&seed) / RAND_MAX;
	float cum = 0.0f;
	for (int i = 0; i < top_k; ++i) {
		cum += probs[i];
		if (r < cum) {
			int selected = indices[i];
			free(logits);
			free(indices);
			free(probs);
			return selected;
		}
	}

	// Fallback
	int fallback = indices[top_k - 1];
	free(logits);
	free(indices);
	free(probs);
	return fallback;
}
