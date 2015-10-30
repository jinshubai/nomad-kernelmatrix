#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <mpi.h>
//#include <tbb/scalable_allocator.h>
//#include <tbb/cache_aligned_allocator.h>
//#include <tbb/concurrent_queue.h>
//#include <tbb/concurrent_vector.h>
//#include <tbb/atomic.h>
//#include <tbb/tbb.h>
//#include <tbb/compat/thread>
#include "Scheduler.h"


Model::Model(int lc_l, int glb_l)
{
	this->local_l = lc_l;
	this->global_l = glb_l;
	Q = new double[local_l*global_l];
	
	/*Q = new double*[lc_l];
	for(int i=0;i<lc_l;i++)
	Q[i] = new double[glb_l];*/
}


Model::~Model()
{
	delete[] Q;
	/*
	for(int i=0;i<lc_l;i++)
	delete[] Q[i];
	delete[] Q;
	*/
}

 Scheduler::Scheduler(const struct problem *prob, struct nomad_parameter *param)
 {
	 int i, rank_id;
	 this->l = prob->l;
	 this->prob = prob;
	 this->nr_ranks = param->nr_ranks;
	 //this->rank_ids = new int[nr_ranks];
	 this->l_start_ptr = new int[nr_ranks];
	 this->each_l = new int[nr_ranks];
	 this->lengthOffeature_node = prob->lengthOffeature_node;
	 this->sending_count = new int[nr_ranks];
	 this->receiving_count = new int[nr_ranks];
	 //this->cp_x = new orig_feature_node[l];
 
	 //compute l_start_ptr
	 MPI_Allgather(&l, 1, MPI_INT, each_l, 1, MPI_INT, MPI_COMM_WORLD);
	 //rank_ids[0] = 0;
	 l_start_ptr[0] = 0;
	 for(i=1;i<nr_ranks;i++)//0~2,3~5,6~8.9~11
	 {
		//rank_ids[i] = i;
		 l_start_ptr[i] = l_start_ptr[i-1] + each_l[i-1];
	 }

	 //generate a copy for each feature node
	 for(i=0;i<l;i++)
	 {
		 //DataPool cp_x;
		 //cp_x.first = true;
		 //cp_x.length = lengthOffeature_node[i];
		 //MPI_Comm_rank(MPI_COMM_WORLD, &cp_x.ini_rank);
		 //MPI_Comm_rank(MPI_COMM_WORLD, &cp_x.cur_rank);
		 //cp_x.global_index = i+l_start_ptr[cp_x.cur_rank];
		 //cp_x.x = callocator<feature_node>().allocate(cp_x.length);
		 //memcpy(cp_x.x, prob->x[i],(size_t)cp_x.length*sizeof(feature_node));

		 DataPool *cp_x = callocator<DataPool>().allocate(1);
		 cp_x->first = true;
		 cp_x->length = lengthOffeature_node[i];
		 MPI_Comm_rank(MPI_COMM_WORLD, &cp_x->ini_rank);
		 MPI_Comm_rank(MPI_COMM_WORLD, &cp_x->cur_rank);
		 cp_x->global_index = i + l_start_ptr[cp_x->cur_rank];
		 cp_x->x = callocator<feature_node>().allocate(cp_x->length);
		 memcpy(cp_x->x, prob->x[i],(size_t)cp_x->length*sizeof(feature_node));
		 push(cp_x);
	 }

	 if(nr_ranks>1)
	 {
		 for(rank_id=0;rank_id<nr_ranks;rank_id++)
		 {
			 sending_count[rank_id] = 0;
			 for(i=0;i<nr_ranks;i++)
			 {
				 if((rank_id+1)%nr_ranks != i)
				 {
					 sending_count[rank_id] += each_l[i];
				 }
			 }
			 receiving_count[(rank_id+1)%nr_ranks] = sending_count[rank_id];
		 }
	 }
 }

 Scheduler::~Scheduler()
 {
	 //int i;
	 //delete[] rank_ids;
	 delete[] l_start_ptr;
	 delete[] each_l;
	 delete[] sending_count;
	 delete[] receiving_count;
	 //for(i=0;i<l;i++)
		 //delete[] cp_x[i].x;
	 //delete[] cpx;
	 while(true)
	 {
		 DataPool *cp_x = nullptr;
		 bool successed = queue_.try_pop(cp_x);
		 if(successed)
		 {
			 int lth = cp_x->length;
			 callocator<feature_node>().deallocate(cp_x->x,lth);
			 callocator<DataPool>().destroy(cp_x);
			 callocator<DataPool>().deallocate(cp_x,1);
		 }
		 else
		 {
			 break;
		 }
	 }
 }

 void Scheduler::push(DataPool *cp_x)
 {
	 queue_.push(cp_x);
 }

 DataPool* Scheduler::pop()
 {
	 DataPool *cp_x = nullptr;
	 bool successed = queue_.try_pop(cp_x);
	 if(successed)
		 return cp_x;
	 else
		 return nullptr;
 }