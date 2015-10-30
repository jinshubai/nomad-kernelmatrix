#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <ctype.h>
#include <errno.h>
#include <sched.h>
#include <mutex>
#include <locale.h>
#include <functional> 
#include <unistd.h> 
#include <vector>
#include <thread> 
#include <mpi.h>
#include <omp.h>
#include <condition_variable>
//#include <tbb/tbb.h>
//#include <tbb/compat/thread>
//#include <tbb/scalable_allocator.h>
//#include <tbb/cache_aligned_allocator.h>
//#include <tbb/concurrent_queue.h>
//#include <tbb/concurrent_vector.h>
//#include <tbb/atomic.h>
#include "Scheduler.h"
std::mutex mtx;
typedef tbb::concurrent_queue<DataPool*, callocator<DataPool*>> con_queue;
//typedef std::vector<unsigned char> StreamType;
template <typename T>
using sallocator = tbb::scalable_allocator<T>;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL
#define UNITS_PER_MSG = 100;

//void print_null(const char *s) {}

int mpi_get_rank()
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return rank;	
}

int mpi_get_size()
{
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	return size;	
}

template<typename T>
void mpi_allreduce(T *buf, const int count, MPI_Datatype type, MPI_Op op)
{
  std::vector<T> buf_reduced(count);
	MPI_Allreduce(buf, buf_reduced.data(), count, type, op, MPI_COMM_WORLD);
	for(int i=0;i<count;i++)
		buf[i] = buf_reduced[i];
}


void mpi_exit(const int status)
{
	MPI_Finalize();
	exit(status);
}

void exit_with_help()
{
	printf(
	"Usage: mpirun -n 1 nomad-q [options] training_set_file [model_file]\n"
	"options:\n"
	"-t thread counts (default 4)\n"
	);
	mpi_exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"[rank %d] Wrong input format at line %d\n", mpi_get_rank(), line_num);
	mpi_exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *test_file_name);
void read_problem(const char *filename);
int save_model(const char *model_file_name, const struct Model *model_);
//void free_model_content(struct model *model_ptr);
//void free_and_destroy_model(struct model **model_ptr_ptr);
struct feature_node *x_space;
struct nomad_parameter Param;
struct problem Prob;
//struct model* model_;

int cyclic_loading_rank(int cur_rank, int nr_ranks){///{{{
	 int next_rank;
	 next_rank = (cur_rank+1)%nr_ranks;
	 return next_rank;
 }////}}}

void nomad(const problem *prob, const nomad_parameter *param, const Model *model_, Scheduler *scheduler)
{
	int cur_rank = mpi_get_rank(); 
	//basic parameters
	int l = prob->l;
	int thread_count = param->thread_count;
	printf("thread_count=%d\n",thread_count);
	fflush(stdout);
	int nr_ranks = param->nr_ranks;
	int global_l = model_->global_l;
	double *Q = model_->Q;
	int *sending_count = scheduler->sending_count;
	int *receiving_count = scheduler->receiving_count;
	//atomic parameters
	atomic<int> count_setup_threads;
	count_setup_threads = 0;
	atomic<int> send_count;
	send_count = 0;
	atomic<int> receive_count;
	receive_count = 0;
	atomic<int> compute_count;
	compute_count = 0;
	atomic<bool> flag_send_ready;
	flag_send_ready = false;
	atomic<bool> flag_receive_ready;
	flag_receive_ready = false;
	//auxiliary buffer
	//std::vector<unsigned char> send_buf;
	//std::vector<unsigned char> receive_buf;

	//setup job_queues for the task
	//each thread owns each queue with corresponding access
	con_queue *job_queues = callocator<con_queue>().allocate(thread_count);
	for(int i=0;i<thread_count;i++)
			callocator<con_queue>().construct(job_queues + i);
	// a sending queue
	con_queue send_queue;
	//initialize job_queues.
	int interval_count = (int)ceil((double)l/thread_count);
	int thread_id = 0;
	for(int i=0;i<prob->l;i++)
	{
		DataPool *cp_x = nullptr;
		cp_x = scheduler->pop();
		if((i!=0)&&(i%interval_count==0))
			thread_id++;
		job_queues[thread_id].push(cp_x);
	}
/************************************************/
auto Qi = [&](struct DataPool *cp_x)->void{//{{{
	int i = 0;
	for(i=0;i<l;i++)
	{
		feature_node *s = prob->x[i];
		feature_node *t = cp_x->x;
		double sum = 0;
		while(s->index!=-1 && t->index!=-1)
		{
			if(s->index==t->index)
			{
				sum += s->value*t->value;
				 s++;
				 t++;}
			else
			{
				if(s->index > t->index)
					 t++;
				else
					s++;
				}
			}
		Q[cp_x->global_index+i*global_l] = sum;
//printf("Q[%d]=%f\n",cp_x->global_index+i*global_l, Q[cp_x->global_index+i*global_l]);
	}

};///}}}

/*************************************************/
auto computer_fun = [&](int thread_id)->void{ ///{{{

	 count_setup_threads++;
	 //printf("count_setup_threads=%d\n", count_setup_threads);
	 while(count_setup_threads < thread_count){
		 std::this_thread::yield();}
	 while(true)
	{
		if(compute_count == global_l)
			break;
		DataPool *cp_x = nullptr;
		bool pop_successed = job_queues[thread_id].try_pop(cp_x);
		if(pop_successed)
		{
			if(cp_x->first)
			{
				Qi(cp_x);
				//printf("cp_x->global_index = %d\n", cp_x->global_index);
				compute_count++;

				if(mpi_get_rank()==2){
				printf("compute_count=%d\n", compute_count);
				}

				if(nr_ranks == 1)
				{
					int lth = cp_x->length;
					//printf("lth=%d\n",lth);
					callocator<feature_node>().deallocate(cp_x->x, lth);
					callocator<DataPool>().destroy(cp_x);
					callocator<DataPool>().deallocate(cp_x,1);
					//std::this_thread::yield();
				}
				else
				{
					cp_x->first = false;
					send_queue.push(cp_x);
					flag_send_ready = true;
				}
			}
			else
			{
				Qi(cp_x);
				compute_count++;
				
				if(mpi_get_rank()==2){
					printf("compute_count=%d\n", compute_count);
				}
				cp_x->cur_rank = mpi_get_rank();
				int next_rank = cyclic_loading_rank(cp_x->cur_rank, nr_ranks);
				if(next_rank == cp_x->ini_rank)
				{
					int lth = cp_x->length; 
					callocator<feature_node>().deallocate(cp_x->x, lth);
					callocator<DataPool>().destroy(cp_x);
					callocator<DataPool>().deallocate(cp_x,1);
					//std::this_thread::yield();
				}
				else
				{
					send_queue.push(cp_x);
				}
			}
		} 
	}
	return;
 };///}}}
/*************************************************/
auto sender_fun = [&]()->void{///{{{
	while(flag_send_ready == false)
	{
		std::this_thread::yield();
	}
	int lth;
	int msg_bytenum;
	while(true)
	{
		if(send_count == sending_count[mpi_get_rank()])
			break;
		DataPool *cp_x = nullptr;
		bool pop_successed = send_queue.try_pop(cp_x);
		if(pop_successed)
		{
			int next_rank = cyclic_loading_rank(cp_x->cur_rank, nr_ranks);
			if(next_rank == cp_x->ini_rank)
			{
				lth = cp_x->length; 
				callocator<feature_node>().deallocate(cp_x->x, lth);
				callocator<DataPool>().destroy(cp_x);
				callocator<DataPool>().deallocate(cp_x,1);
				//std::this_thread::yield();
			}
			else
			{
				lth = cp_x->length; 
				msg_bytenum = sizeof(bool)+4*sizeof(int)+lth*sizeof(feature_node);
				char *send_message = sallocator<char>().allocate(msg_bytenum);
				*(reinterpret_cast<bool *>(send_message)) = cp_x->first;
				*(reinterpret_cast<int *>(send_message + sizeof(bool))) = cp_x->length;
				*(reinterpret_cast<int *>(send_message + sizeof(bool) + sizeof(int))) = cp_x->ini_rank;
				*(reinterpret_cast<int *>(send_message + sizeof(bool) + 2*sizeof(int))) = cp_x->cur_rank;
				*(reinterpret_cast<int *>(send_message + sizeof(bool) + 3*sizeof(int))) = cp_x->global_index;
				feature_node *dest = reinterpret_cast<feature_node *>(send_message + sizeof(bool) + 4*sizeof(int));
				std::copy(cp_x->x, cp_x->x + lth, dest);
				flag_receive_ready = true;
				MPI_Ssend(send_message, msg_bytenum, MPI_CHAR, next_rank, 1, MPI_COMM_WORLD);
				//destroying
				callocator<feature_node>().deallocate(cp_x->x, lth);
				callocator<DataPool>().destroy(cp_x);
				callocator<DataPool>().deallocate(cp_x,1);
				//record the sending count
				send_count++;
//if(mpi_get_rank()==1)
//{
//				printf("send_count=%d\n", send_count);
//}
				sallocator<char>().deallocate(send_message, msg_bytenum);
			}
		}
	}
	return;
};///}}}
/*************************************************/
auto receiver_fun = [&]()->void{///{{{

	while(flag_receive_ready == false)
	{
		std::this_thread::yield();
	}
	int flag = 0;
	int src_rank;
	int lth;
	MPI_Status status;
	while(true)
	{
		if(receive_count == receiving_count[mpi_get_rank()])
			break;
		MPI_Iprobe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flag, &status);
		if(flag == 0)
		{
			std::this_thread::yield();
		}
		else
		{
			src_rank = status.MPI_SOURCE;
			int msg_size = 0; 
			MPI_Get_count(&status, MPI_CHAR, &msg_size);
			char *recv_message = sallocator<char>().allocate(msg_size);
			MPI_Recv(recv_message, msg_size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &status);
			//recovering
			DataPool *cp_x = callocator<DataPool>().allocate(1);
			cp_x->first = *(reinterpret_cast<bool *>(recv_message));
			cp_x->length = *(reinterpret_cast<int *>(recv_message + sizeof(bool)));
			cp_x->ini_rank = *(reinterpret_cast<int *>(recv_message + sizeof(bool) + sizeof(int)));
			cp_x->cur_rank = *(reinterpret_cast<int *>(recv_message + sizeof(bool) + 2*sizeof(int)));
			cp_x->global_index = *(reinterpret_cast<int *>(recv_message + sizeof(bool) + 3*sizeof(int)));
			feature_node *dest = reinterpret_cast<feature_node *>(recv_message + sizeof(bool) + 4*sizeof(int));
			//please notice that the approach to recover cp_x->x
			lth = cp_x->length;
			cp_x->x = callocator<feature_node>().allocate(cp_x->length);
			memcpy(cp_x->x, dest, (size_t)sizeof(feature_node)*lth);
			sallocator<char>().deallocate(recv_message, msg_size); 
			//change the current rank of received cp_x

			//push an item to the job_queue who has the smallest number of items.
			//In doing so, the dynamic loading balancing can be achieved.	
			int smallest_items_thread_id = 0;	
			auto smallest_items = job_queues[0].unsafe_size();	
			for(int i=1;i<thread_count;i++)	
			{
				auto tmp = job_queues[i].unsafe_size();		
				if(tmp < smallest_items)		
				{			
					smallest_items_thread_id = i;			
					smallest_items = tmp;		
				}	
			}
			job_queues[smallest_items_thread_id].push(cp_x);
			receive_count++;
//if(mpi_get_rank()==1)
//{
//				printf("receive_count=%d\n", receive_count);
//}
		}
	}
	return;
};///}}}
/*************************************************/
	//create some functional threads
	std::vector<std::thread> computers;
	std::thread *sender = nullptr;
	std::thread *receiver = nullptr;
	//multi-threads are applied to do the jobs via the job_queues, send_queue etc.
    for (int i=0; i < thread_count; i++) {
		computers.push_back(std::thread(computer_fun, i));
    }
	if(nr_ranks>1)
	{
		sender = new std::thread(sender_fun);
		receiver = new std::thread(receiver_fun);
	}
	//wait until data loading and initialization that have done.
	//wait the computers doing the job, this is main thread to do this job
	while(count_setup_threads < thread_count){
			std::this_thread::yield();
		}
	printf("Start Jobs!!\n");
	fflush(stdout);
	tbb::tick_count start_time = tbb::tick_count::now();
	while (true){
		if(nr_ranks==1)
		{
			if(compute_count==global_l)
				break;
		}
		else
		{			
		if((compute_count==global_l)&&(send_count==sending_count[cur_rank])
			&&(receive_count==receiving_count[cur_rank]))
			break;
		}
	}
	double elapsed_seconds = (tbb::tick_count::now() - start_time).seconds();
	printf("All done, the costed time is %f Seconds\n", elapsed_seconds);
	fflush(stdout);
	printf("Now free memory!\n");
	fflush(stdout);
	callocator<con_queue>().deallocate(job_queues, thread_count); 
	// thread join
	for(auto &th: computers)
		th.join();
	if(nr_ranks > 1)
	{
		sender->join();
		receiver->join();
		delete sender;
		delete receiver;
	}
}

int main(int argc, char **argv)
{
	//Below lists the arguments for MPI;
	//MPI_Status status;
	int threadprovided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadprovided);
	if(threadprovided != MPI_THREAD_MULTIPLE)
	{
		printf("MPI multiple thread isn't provided!\n");
		fflush(stdout);
		mpi_exit(1);
	}
	int cur_rank = mpi_get_rank();
	int nr_ranks = mpi_get_size();
	Param.nr_ranks = nr_ranks;

	char hostname[1024];
	int hostname_len;
	MPI_Get_processor_name(hostname, &hostname_len);
    printf("processor name: %s, number of processed: %d, rank: %d\n", hostname, nr_ranks, cur_rank);

	/* Set the gloal arguments*/
	int global_l, global_n;	
	char input_file_name[1024];
	char model_file_name[1024];
	//const char *error_msg;
	parse_command_line(argc, argv, input_file_name, model_file_name);
	read_problem(input_file_name);

	//distributed code
	global_l = Prob.l;
	global_n = Prob.n;

	if(nr_ranks > 1)
	{
		mpi_allreduce(&global_l, 1, MPI_INT, MPI_SUM);//MPI_INT :int;MPI_SUM:sum
		mpi_allreduce(&global_n, 1, MPI_INT, MPI_MAX);//MPI_MAX:max
		Prob.n = global_n;
	}

	if(cur_rank==0)
		printf("#instance = %d, #feature = %d\n", global_l, global_n);

	//construct model for each machine
	struct Model *model_ = nullptr;
	model_ = new Model(Prob.l, global_l);
	
	//construct a scheduler for processing
	Scheduler *scheduler = nullptr;
	scheduler = new Scheduler(&Prob, &Param);

	//please notice that this is our jobs
	nomad(&Prob, &Param, model_, scheduler);
	
	/*if(save_model(model_file_name, model_))
	{
		fprintf(stderr,"[rank %d] can't save model to file %s\n",mpi_get_rank(), model_file_name);
		mpi_exit(1);
	}*/
	delete model_;
	delete scheduler;
	//free_and_destroy_model(&model_);
	free(Prob.y);
	free(Prob.x);
	free(Prob.query);
	free(x_space);
	free(Prob.lengthOffeature_node);
	free(line);

	MPI_Finalize();
	return 0;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	//void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	Param.thread_count = 4;//default thread count

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 't':
				Param.thread_count = atoi(argv[i]);
				break;

			/*case 'q':
				print_func = &print_null;
				i--;
				break;*/

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	//set_print_string_function(print_func);

	// determine filenames
	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"/home/jing/dis_data/%s.model",p);
	}
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	long int elements, j, k;

	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	Prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		Prob.l++;
	}
	rewind(fp);

	Prob.lengthOffeature_node = Malloc(int,Prob.l);
	Prob.y = Malloc(double,Prob.l);
	Prob.x = Malloc(struct feature_node *,Prob.l);
	Prob.query = Malloc(int,Prob.l);
	x_space = Malloc(struct feature_node,elements+Prob.l);
	max_index = 0;
	j=0;
	k=0;
	for(i=0;i<Prob.l;i++)
	{
		Prob.query[i] = 0;
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		Prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		Prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			if (!strcmp(idx,"qid"))
			{
				errno = 0;
				Prob.query[i] = (int) strtol(val, &endptr,10);
				if(endptr == val || errno !=0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(i+1);
			}
			else
			{
				errno = 0;
				x_space[j].index = (int) strtol(idx,&endptr,10);
				if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
					exit_input_error(i+1);
				else
					inst_max_index = x_space[j].index;

				errno = 0;
				x_space[j].value = strtod(val,&endptr);
				if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(i+1);

				++j;
			}
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		x_space[j++].index = -1;
		Prob.lengthOffeature_node[i] = (int)(j-k);
		k = j;
	}
	Prob.n=max_index;
	fclose(fp);
}

int save_model(const char *model_file_name, const Model *model_)
{
	int i;
	int local_l = model_->local_l;
	int global_l = model_->global_l;

	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	fprintf(fp, "local_count: %d\n", model_->local_l);
	fprintf(fp, "globa_count: %d\n", model_->global_l);

	for(i=0; i<local_l; i++)
	{
		int j;
		for(j=0; j<global_l; j++)
			fprintf(fp, "%.16g(%d,%d) ", model_->Q[i*global_l+j],i,j);
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

/*void free_model_content(struct model *model_ptr)
{
	if(model_ptr->Q != NULL)
		free(model_ptr->Q);
}*/

/*void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}*/
