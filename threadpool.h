#ifndef __THREADPOOL_H__
#define __THREADPOOL_H__

#include <pthread.h>
#include <stdatomic.h>
#include "config.h"

typedef void (*task_func_t)(void *arg);

typedef struct {
	task_func_t func;
	void *arg;
} task_t;

typedef struct {
	pthread_t threads[MAX_THREADS];
	task_t tasks[MAX_TASKS];
	atomic_int task_queue_count; // Renamed for clarity
	atomic_int task_head;
	atomic_int task_tail;
	pthread_mutex_t mutex;
	pthread_cond_t cond_not_empty;
	pthread_cond_t cond_not_full;
	atomic_bool running;
	int num_threads;

	// --- FIX: Add state for proper waiting ---
	atomic_int active_tasks;      // Tasks submitted but not yet finished
	pthread_cond_t cond_all_done; // To signal when active_tasks is zero
} thread_pool_t;

extern thread_pool_t *thread_pool;

extern thread_pool_t *thread_pool_create(int num_threads);
extern void thread_pool_submit(thread_pool_t *pool, task_func_t func, void *arg);
extern void thread_pool_wait(thread_pool_t *pool);
extern void thread_pool_destroy(thread_pool_t *pool);

#endif
