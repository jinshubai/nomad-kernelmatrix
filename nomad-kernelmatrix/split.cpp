#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <ctype.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include "split.h"
#define INF HUGE_VAL
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
void exit_main()
{
     printf(
		 "Usage:./split -p [integer] -e [decimal] input_file\n"
		 "p: number of computing nodes (machines)\n"
		 "e: allowed error to keep balance (a decimal, e.g., e = 0.001)\n"
		 "input_file: name of the input file\n");
	 exit(1);     
}
void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n",line_num);
	exit(1);
}
static int compare_query(const void *a, const void *b)
{
	int *ia = (int *)a;
	int *ib = (int *)b;
	if(*ia > *ib)
		return -1;
	if(*ia < *ib)
		return 1;
	return 0;
}
static char *line = NULL;
static int max_line_len;
static char *readline(FILE *input)
{
	int len;
	if(fgets(line,max_line_len,input)==NULL)
		return NULL;
	while(strrchr(line,'\n')==NULL)
	{
		max_line_len *=2;
		line = (char*)realloc(line, max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len, max_line_len-len, input)==NULL)
			break;
	}
	return line;
}
void statistic_queries(char *input_file, int *query, int l)
{
	char *endptr;
	char *idx, *val, *label;
	//double y = 0.0;
	FILE *fp = fopen(input_file,"r");
	for(int i=0;i<l;i++)
	{
		readline(fp);
		label = strtok(line," \t\n");
		if(label == NULL)
			exit_input_error(i+1);
		//double y = strtod(label,&endptr);
		strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);
        idx = strtok(NULL,":");
		val = strtok(NULL," \t");
		if(val == NULL)
			exit_input_error(i+1);
		if(!strcmp(idx,"qid"))
		{
			errno = 0;
			query[i] = (int) strtol(val, &endptr, 10);
			if(endptr == val || errno !=0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);
		}		
	}
	rewind(fp);
	fclose(fp);
}
void split(char *input_file, int l, int machines, int nr_query, struct Query_Machine *q_machine, int *query)
{
	int machine_id = 0;
	//double y = 0.0;
	int len = 0;
	FILE *fp = fopen(input_file,"r");
	char *idx, *val, *endptr;
	char *label;
	char **out_file = (char**)malloc(sizeof(char*)*machines);
	for(int i=0;i<machines;i++)
		out_file[i] = (char*)malloc(sizeof(char)*1024);
    FILE **f = (FILE**)malloc(sizeof(FILE*)*machines);
    
    if(mkdir("temp_dir",0777)==0)
    {
        printf("Directory was successfully created!\n");
    }
    else
    {
        printf("Directory has existed!!\n");
    }
    
    for(int i=0;i<machines;i++)
	{
		sprintf(out_file[i],"temp_dir/%s.%d",input_file,i);
		f[i] = fopen(out_file[i],"w");
	}

	char *copy_line = (char*)malloc(sizeof(char)*2048);
	for(int j=0;j<l;j++)
	{
		readline(fp);		
		len = (int)strlen(line);
		//printf("len=%d for line:%d\n",len,j+1);
		if(len > 2048)
		{
			copy_line = (char*)realloc(copy_line,len*sizeof(char));
		}
		strcpy(copy_line,line);
		//printf("copy_line:%s",copy_line);
		//printf("line:%s",line);
		label = strtok(line, " \t\n");
		if(label == NULL)
			exit_input_error(j+1);
		//double y = strtod(label, &endptr);
		strtod(label, &endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(j+1);
		idx = strtok(NULL,":");
		val = strtok(NULL, " \t");
		if(val == NULL)
			exit_input_error(j+1);
		if(!strcmp(idx,"qid"))
		{
			errno = 0;
			query[j] = (int)strtol(val, &endptr, 10);
			if(endptr == val || errno !=0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(j+1);
		}
		for(int i=0;i<nr_query;i++)
		{
			if(query[j] == q_machine[i].query)
			{
				machine_id = q_machine[i].machine_id;
				break;
			}
		}
		fprintf(f[machine_id],"%s",copy_line);

	}
	free(copy_line);
	for(int i=0;i<machines;i++)
		free(out_file[i]);
	free(out_file);
	for(int i=0;i<machines;i++)
		fclose(f[i]);
	rewind(fp);
	fclose(fp);
}
int main(int argc, char **argv)
{
	char *input_file = NULL;
	input_file = (char*)malloc(sizeof(char)*1024);
	int machines = 1;
	double epsilon = 0.001;
	if(argc != 6) exit_main();
	for(int i=1;i<argc-1;i++)
	{
		if(argv[i][0]!='-')
		{
			exit_main();
		}
		i++;
		switch(argv[i-1][1])
		{
		case 'p':
			machines = atoi(argv[i]);
			break;
		case 'e':
			epsilon = atof(argv[i]);
			break;
		default:
			fprintf(stderr,"unknow options:-%s\n",argv[i-1]);
			exit_main();
			break;
		}
	}
	strcpy(input_file, argv[argc-1]);
	FILE *fp = fopen(input_file,"r");
    printf("Split the data......\n");
	int *query = NULL;
	int k = 0;
	int nr_query = 1;
	int l = 0;
	struct Query_Info *q_info = NULL;
	int *nr_in_machine = NULL;
	struct Query_Machine *query_machine = NULL;
	nr_in_machine = (int*)malloc(sizeof(int)*machines);
	for(int i=0;i<machines;i++)
		nr_in_machine[i] = 0;
	if(fp==NULL)
	{
		fprintf(stderr, "can't open input file %s\n",input_file);
		exit(1);
	}
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		l++;
	}
	rewind(fp);
	fclose(fp);
	//printf("l=%d\n",l);
	//exit(1);
	query = (int*)malloc(sizeof(int)*l);
	statistic_queries(input_file, query, l);
	qsort(query, l, sizeof(int), compare_query);// in decending order
	//for(int i=0;i<l;i++)
		//printf("query:%d\n",query[i]);
	//exit(1);
	for(int j=1;j<l;j++)
	{
		if(query[j] != query[j-1])
			nr_query++;
	}
	//printf("nr_query=%d",nr_query);
	//exit(1);
    //
	q_info = (struct Query_Info *)malloc(sizeof(Query_Info)*nr_query);
	q_info[0].query = query[0];
	for(int j=0;j<nr_query;j++)
	{
		q_info[j].num = 0;
		q_info[j].selected = false;
		q_info[j].machine_id = INF;
	}
	// calculate the number of instances for each query.
	for(int j=0;j<l;j++)
	{
		if(query[j]==q_info[k].query)
		{
			q_info[k].num++;
		}
		else
		{
			k++;
			q_info[k].query = query[j];
			q_info[k].num++;
		}
	}
	//address the imbalance issue. i.e., different computing nodes will get almost
    //the same number of instances.
	//solve_imbalance_issue(q_info, nr_query, machines, l, epsilon, nr_in_machine);
    printf("Slove the imbalance issue.\n");
	query_machine = (Query_Machine*)malloc(sizeof(Query_Machine)*nr_query);
	address_imbalance_doublelist(q_info, nr_query, machines, l, epsilon, nr_in_machine, query_machine);
	//for(int i=0;i<machines;i++)
		//printf("nr_in_machine[%d]=%d\n",i,nr_in_machine[i]);
	//exit(1);
	//for(int i=0;i<nr_query;i++)
		//printf("query:%d->machined_id:%d\n",q_info[i].query,q_info[i].machine_id);
    //exit(1);

	double average = (double) l/machines;	 
	 for(int i=0;i<machines;i++)
	 {
		 printf("Instance Count: %d in Machine %d* ",nr_in_machine[i], i);
		 printf("Difference:%d* ", nr_in_machine[i]-(int)average);
         printf("Difference Rate:%f\n",((double)nr_in_machine[i]-average)/average*100.0);
     }


	//split the input file
	split(input_file, l, machines, nr_query, query_machine, query);
	printf("The splited data have been completed!!\n");
    //free the assigned memory
	//system("cd temp_dir/");
    //system("ls");
    //system("cd ..");
    //system("rm -r temp_dir");
    free(q_info);
    free(nr_in_machine);
    free(query_machine);
	free(query);
	free(line);
	free(input_file);
	return 0;
}
