#ifndef _SPLIT_H_
#define _SPLIT_H_
struct Query_Info
{
       int query;
       int num;
       bool selected;
       int machine_id;
};

struct Query_Machine
{
	int query;
	int machine_id;
};

typedef struct Node{// define the node of double direction list
	struct Query_Info q_info;
	Node *next;
	Node *prior;
}Node, *Dlist;

Dlist creat_doublelist(struct Query_Info *q_info, int nr_query)
{
	struct Node *head, *q, *p;
	head = (struct Node*)malloc(sizeof(Node));
	head->q_info.query =  q_info[0].query;
	head->q_info.num = q_info[0].num;
	head->q_info.selected = q_info[0].selected;
	head->q_info.machine_id = q_info[0].machine_id;
	head->prior = NULL;
	head->next = NULL;
	q = head;
	for(int i=1;i<nr_query;i++)
	{
		p = (struct Node*)malloc(sizeof(Node));
		p->q_info.query =  q_info[i].query;
		p->q_info.num = q_info[i].num;
		p->q_info.selected = q_info[i].selected;
		p->q_info.machine_id = q_info[i].machine_id;

		p->prior = q;
		q->next = p;
		p->next = NULL;
		q = p;
	}

	return head;
}

Dlist delt_node(Dlist head, Dlist expried_node)
{// Notice that expried_node->q_info.selected == true should be true
	if(head == expried_node)
	{
		if(head->next != NULL)// if expried_node is the head of this double list
		{
			head = expried_node->next;
			head->prior = NULL;
			free(expried_node);
			return head;
		}
		else// if the double list has only one node or this node is the last one
		{
			free(head);//i.e., free(expried_node);
			return NULL;
		}
	}
	else
	{
		if(expried_node->next != NULL)// if expried_node is not the last node of this double list
		{
			expried_node->prior->next = expried_node->next;
			expried_node->next->prior = expried_node->prior;
			free(expried_node);
			return head;
		}
		else
		{
			expried_node->prior->next = NULL;
			free(expried_node);
			return head;
		}
	}
}
void address_imbalance_doublelist(struct Query_Info *q_info, int nr_query, int machines, int l, double epsilon, int *nr_in_machine, struct Query_Machine *query_machine)
{
	int i, k;
	int nr_epsilon;
	double average = (double) l/machines;
	nr_epsilon = (int)ceil(epsilon*average);
	printf("The average number:%d\n",(int)average);
	printf("The allowed maxmum error:%d\n",nr_epsilon);

	//create the double direction list
	Dlist List = creat_doublelist(q_info, nr_query);
	Dlist p;
	struct Query_Machine *q_machine = &query_machine[0];
	//int j = 0;

	//address the imbalance issue
	//the first phase 
	for(i=0;i<machines-1;i++)
	{
		p = List;
		while(p)
		{
			Dlist q;
			if(p->q_info.selected == false)
			{
				if(nr_in_machine[i] == 0)
				{
					nr_in_machine[i] += p->q_info.num;
					p->q_info.selected = true;
					p->q_info.machine_id = i;
					q_machine->machine_id = i;
					q_machine->query = p->q_info.query;
					q_machine++;
					q = p;
					p = p->next;
					List = delt_node(List,q);
				}
				else
				{
					nr_in_machine[i] += p->q_info.num;
					k = nr_in_machine[i]-(int)average;

					if(k < 0)
					{
						p->q_info.selected = true;
						p->q_info.machine_id = i;
						q_machine->machine_id = i;
						q_machine->query = p->q_info.query;
						q_machine++;
						q = p;
						p = p->next;
						List = delt_node(List,q);
					}

					if(k>=0 && k<=nr_epsilon)
					{
						p->q_info.selected = true;
						p->q_info.machine_id = i;
						q_machine->machine_id = i;
						q_machine->query = p->q_info.query;
						q_machine++;
						q = p;
						p = p->next;
						List = delt_node(List,q);
						break;
					} 

					if(k>nr_epsilon)
					{
						nr_in_machine[i] -= p->q_info.num;
						p = p->next;
					}
				}
			}

		}
	}
	//the second phase
	while(List)
	{
		Dlist q = List;
		if(q->q_info.selected == false)
		{
			q->q_info.selected = true;
			q->q_info.machine_id = machines-1;
			q_machine->machine_id = machines-1;
			q_machine->query = q->q_info.query;
			q_machine++;
			nr_in_machine[machines-1] += q->q_info.num;
			List = delt_node(List, q);
		}

	}
}



void solve_imbalance_issue(struct Query_Info *q_info, int nr_query, int machines, int l, double epsilon, int *nr_in_machine)
{
	int i, j, k;
	int nr_epsilon;
	double average = (double) l/machines;
	nr_epsilon = (int)ceil(epsilon*average);
	//printf("The average number:%d\n",(int)average);
    //printf("The allowed maxmum error:%d\n",nr_epsilon);
	//The fist phase
	for (i=0;i<machines-1;i++)
	{
		for(j=0;j<nr_query;j++)
		{
			if(q_info[j].selected==false)
			{
				if(nr_in_machine[i]==0)
				{
					nr_in_machine[i] += q_info[j].num;
					q_info[j].selected = true;
					q_info[j].machine_id = i;
				}
				else
				{
					nr_in_machine[i] += q_info[j].num;
					k = nr_in_machine[i]-(int)average;
					if(k < 0)
					{
						q_info[j].selected = true;
						q_info[j].machine_id = i;
					}
					if(k>=0 && k<=nr_epsilon)
					{
						q_info[j].selected = true;
						q_info[j].machine_id = i;
						break;
					} 
					if(k>nr_epsilon)
					{
						nr_in_machine[i] -= q_info[j].num;
						continue;
					}					
				}
			}	
		}
	}
	// The second phase
	for(j=0;j<nr_query;j++)
	{
		if(q_info[j].selected == false)
		{
			nr_in_machine[machines-1] += q_info[j].num;
			q_info[j].selected = true;
			q_info[j].machine_id = machines-1;
		}
	}
}
#endif
