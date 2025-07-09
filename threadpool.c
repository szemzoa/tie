#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include "threadpool.h"

thread_pool_t *thread_pool;

static void *thread_pool_worker(void *arg);

// Initialize thread pool
thread_pool_t *thread_pool_create(int num_threads)
{
	if (num_threads <= 0 || num_threads > MAX_THREADS) {
		num_threads = sysconf(_SC_NPROCESSORS_ONLN); // Use CPU core count
		if (num_threads > MAX_THREADS)
			num_threads = MAX_THREADS;
	}

	thread_pool_t *pool = malloc(sizeof(thread_pool_t));
	if (!pool)
		return NULL;

	pool->task_queue_count = 0;
	pool->task_head		   = 0;
	pool->task_tail		   = 0;
	pool->running		   = true;
	pool->num_threads	   = num_threads;
	// --- FIX: Initialize new members ---
	pool->active_tasks = 0;

	pthread_mutex_init(&pool->mutex, NULL);
	pthread_cond_init(&pool->cond_not_empty, NULL);
	pthread_cond_init(&pool->cond_not_full, NULL);
	// --- FIX: Initialize new condition variable ---
	pthread_cond_init(&pool->cond_all_done, NULL);

	for (int i = 0; i < num_threads; i++) {
		pthread_create(&pool->threads[i], NULL, thread_pool_worker, pool);
	}

	return pool;
}

// Worker thread function
static void *thread_pool_worker(void *arg)
{
	thread_pool_t *pool = (thread_pool_t *)arg;

	while (atomic_load(&pool->running)) {
		pthread_mutex_lock(&pool->mutex);

		// Wait for tasks
		while (pool->task_queue_count == 0 && atomic_load(&pool->running)) {
			pthread_cond_wait(&pool->cond_not_empty, &pool->mutex);
		}

		if (!atomic_load(&pool->running) && pool->task_queue_count == 0) {
			pthread_mutex_unlock(&pool->mutex);
			break;
		}

		// Get task
		task_t task		= pool->tasks[pool->task_head];
		pool->task_head = (pool->task_head + 1) % MAX_TASKS;
		atomic_fetch_sub(&pool->task_queue_count, 1);

		pthread_cond_signal(&pool->cond_not_full);
		pthread_mutex_unlock(&pool->mutex);

		// Execute task
		if (task.func) {
			task.func(task.arg);
		}

		// --- FIX: Signal task completion ---
		pthread_mutex_lock(&pool->mutex);
		atomic_fetch_sub(&pool->active_tasks, 1);
		if (atomic_load(&pool->active_tasks) == 0) {
			// If this was the last active task, signal the waiting main thread
			pthread_cond_signal(&pool->cond_all_done);
		}
		pthread_mutex_unlock(&pool->mutex);
	}

	return NULL;
}

// Submit task
void thread_pool_submit(thread_pool_t *pool, task_func_t func, void *arg)
{
	pthread_mutex_lock(&pool->mutex);

	// Wait if queue is full
	while (pool->task_queue_count == MAX_TASKS) {
		pthread_cond_wait(&pool->cond_not_full, &pool->mutex);
	}

	// Add task
	pool->tasks[pool->task_tail] = (task_t){func, arg};
	pool->task_tail				 = (pool->task_tail + 1) % MAX_TASKS;
	atomic_fetch_add(&pool->task_queue_count, 1);

	// --- FIX: Increment active task counter ---
	atomic_fetch_add(&pool->active_tasks, 1);

	pthread_cond_signal(&pool->cond_not_empty);
	pthread_mutex_unlock(&pool->mutex);
}

// Wait for all tasks to complete
void thread_pool_wait(thread_pool_t *pool)
{
	// --- FIX: Use condition variable for a robust, non-busy wait ---
	pthread_mutex_lock(&pool->mutex);
	while (atomic_load(&pool->active_tasks) > 0) {
		pthread_cond_wait(&pool->cond_all_done, &pool->mutex);
	}
	pthread_mutex_unlock(&pool->mutex);
}

// Destroy thread pool
void thread_pool_destroy(thread_pool_t *pool)
{
	if (!pool)
		return;

	// Wait for any remaining tasks to complete before shutting down
	thread_pool_wait(pool);

	atomic_store(&pool->running, false);
	pthread_mutex_lock(&pool->mutex);
	pthread_cond_broadcast(&pool->cond_not_empty);
	pthread_mutex_unlock(&pool->mutex);

	for (int i = 0; i < pool->num_threads; i++) {
		pthread_join(pool->threads[i], NULL);
	}

	pthread_mutex_destroy(&pool->mutex);
	pthread_cond_destroy(&pool->cond_not_empty);
	pthread_cond_destroy(&pool->cond_not_full);
	// --- FIX: Destroy the new condition variable ---
	pthread_cond_destroy(&pool->cond_all_done);
	free(pool);
}
