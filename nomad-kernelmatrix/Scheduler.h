#ifndef SCHEDULER_H_
#define SCHEDULER_H_
#include <mpi.h>
#include <vector>
#include <tbb/scalable_allocator.h>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>
#include <tbb/atomic.h>
#include <tbb/tbb.h>
#include <tbb/compat/thread>

template <typename T>
using callocator = std::allocator<T>;
//template <typename T>
//using con_queue = tbb::concurrent_queue<T>;
template <typename T>
using atomic = tbb::atomic<T>;
/*template <typename T>
using sallocator = tbb::scalable_allocator<T>;
template <typename T>
using callocator = tbb::cache_aligned_allocator<T>;
using tbb::atomic;*/
using tbb::tick_count;


struct feature_node
{
	int index;
	double value;
};

struct DataPool
{
	bool first;
	int length;
	int ini_rank;
	int cur_rank;
	int global_index;
	struct feature_node *x;
};

struct problem
{
	int l, n;
	int *query;
	double *y;
	struct feature_node **x;
	int *lengthOffeature_node;
};

struct nomad_parameter
{

	int thread_count;// thread count for each computing node
	int nr_ranks;//how many processor are used. one rank, one machine
};

struct Model
{
	Model(int lc_l, int glb_l);
	~Model();
	int local_l;
	int global_l;
	double *Q;
	//double **Q;
};

class Scheduler
{
   public:
	   //struct DataPool *cp_x;
	   Scheduler(const struct problem *prob, struct nomad_parameter *param);
	   ~Scheduler();
	   void push(DataPool *cp_x);
	   DataPool *pop();
	   //void push(DataPool cp_x);
	   //DataPool pop();
	   int *sending_count;
	   int *receiving_count;
   
   private:
	   int l;
	   //int *rank_ids;
	   int *l_start_ptr;
	   const problem *prob;
	   int nr_ranks;
	   int *each_l;
	   int *lengthOffeature_node;
	   tbb::concurrent_queue<DataPool*, callocator<DataPool*>> queue_;
	   //con_queue<DataPool> queue_;
};
#endif