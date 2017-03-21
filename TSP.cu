#include <stdio.h>
#include <math.h>
#include <cutil_inline.h>
#include <omp.h>
#include <sys/time.h>

#define maxnpts 14	//me 14 exoume to beltisto occupancy logw periorismwn ths shared 

#define QUADRO -1	//0 an den yparxei h argh quadro tou diadh, alliws -1

#define BLOCK_SIZE 128	//toulx 256
#define GRID_SIZE 64	//toulaxiston 64

int x[maxnpts], y[maxnpts];
char answer[100];
int npts;
unsigned long int perm=1;

void input(void);
float host_TSP(int *, int *);
float find_final_min(float *, int );
float find_final_min_and_order(float *, int , int *);

__constant__ int d_x[maxnpts];
__constant__ int d_y[maxnpts];
__constant__ unsigned long int d_perm;
__constant__ int d_npts;
__constant__ unsigned long int d_num_a;	//ari8mos epanalhpsewn tou ekswterikou broxou pou 8a keni ka8e nhma
__constant__ int cpu_thread_id;
__constant__ unsigned long int d_perm_per_gpu;

__global__ void dev_TSP(float *d_min, int *d_order);


int main()
{ 
	struct timeval first, second, lapsed;
	struct timezone tzp;
	
	int num_gpus = 0;	// number of CUDA GPUs

    /////////////////////////////////////////////////////////////////
    // determine the number of CUDA capable GPUs
    //
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 1)
    {
       printf("no CUDA capable devices were detected\n");
       return 1;
    }

    /////////////////////////////////////////////////////////////////
    // display CPU and GPU configuration
    //
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", num_gpus);
    for(int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s\n", i, dprop.name);
    }
    printf("---------------------------\n");


	int i;
	input();		//Eisodos dedomenwn
	printf("You have inserted %d points!", npts);
	for(i=0;i<npts;i++){
		printf("(%d,%d)", x[i], y[i]);
	}
	printf("\n");
	for (i=1; i<=npts; perm*=i++);
	
	//****************************************************************
	num_gpus = num_gpus + QUADRO;
	// allocate host memory for the result
	float *h_min;
	int *h_order;
	h_min = (float*)malloc(num_gpus*GRID_SIZE*sizeof(float));
	h_order = (int*)malloc(num_gpus*GRID_SIZE*maxnpts*sizeof(int));
	omp_set_num_threads(num_gpus);
#pragma omp parallel
	{
		if(npts>=8){		//H GPU 8a ypologisei gia 8 shmeia kai panw
			int gpu_id = -1;
			int cpu_thread_id = omp_get_thread_num();	//xarakthristikos ari8mos toy nhmatos tou epeksergasth
			unsigned int num_cpu_threads = omp_get_num_threads();	
			cutilSafeCall(cudaSetDevice(cpu_thread_id));
			cutilSafeCall(cudaGetDevice(&gpu_id));
			printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);
			
			unsigned long int perm_per_gpu = perm/num_gpus;
			if ((unsigned int)cpu_thread_id < perm % num_gpus) perm_per_gpu++;		//gia thn periptwsh pou o ari8mos meta8esewn den diaireitai akribws me ton ari8mo twn kartwn
			
			unsigned long int h_num_a = perm_per_gpu / (BLOCK_SIZE*GRID_SIZE) +1;	//+1 giati h diaresh den einai teleia synh8ws
			
			// allocate device memory
			float *d_min;
			int *d_order;
			cutilSafeCall(cudaMalloc((void**)&d_min, GRID_SIZE*sizeof(float)));
			cutilSafeCall(cudaMalloc((void**)&d_order, GRID_SIZE*maxnpts*sizeof(int)));

			// copy host memory to device
			if(omp_get_thread_num()==0)gettimeofday(&first, &tzp);
			cudaMemcpyToSymbol("d_x", x, npts*sizeof(int));
			cudaMemcpyToSymbol("d_y", y, npts*sizeof(int));
			cudaMemcpyToSymbol("d_npts", &npts, sizeof(int));
			cudaMemcpyToSymbol("d_perm", &perm, sizeof(unsigned long int));

			cudaMemcpyToSymbol("cpu_thread_id", &cpu_thread_id, sizeof(int));	//edw an ebaza px unsigned long int 8a epairna kata thn ektelesh: kernel execution failed, invalid argument
			cudaMemcpyToSymbol("d_perm_per_gpu", &perm_per_gpu, sizeof(unsigned long int));
			cudaMemcpyToSymbol("d_num_a", &h_num_a, sizeof(unsigned long int));
			
			// execute the kernel
			dev_TSP<<<GRID_SIZE, BLOCK_SIZE>>>(d_min,d_order);
			
			// check if kernel execution generated and error
			cutilCheckMsg("Kernel execution failed");
			
			//allocate host memory for the result
			float *ypo_h_min = h_min + cpu_thread_id*GRID_SIZE;
			int *ypo_h_order = h_order + cpu_thread_id*GRID_SIZE*maxnpts;
			cutilSafeCall(cudaMemcpy(ypo_h_min, d_min, GRID_SIZE*sizeof(float), cudaMemcpyDeviceToHost));	//Apo8hkeyw ston h_min to result ths kartas.
			cutilSafeCall(cudaMemcpy(ypo_h_order, d_order, GRID_SIZE*maxnpts*sizeof(int), cudaMemcpyDeviceToHost));	//Apo8hkeyw ston h_min to result ths kartas.
			
			//apeleye8erwsh mnhmhs
			cutilSafeCall(cudaFree(d_min));
			cutilSafeCall(cudaFree(d_order));
			
			if(omp_get_thread_num()==0)gettimeofday(&second, &tzp);
			
		}else printf("H GPU den ypologizei gia ligotero apo 8 shmeia!\n");
	}
	
	if(npts>=8){
		if(first.tv_usec>second.tv_usec){
			second.tv_usec += 1000000;
			second.tv_sec--;
		}
		lapsed.tv_usec = second.tv_usec - first.tv_usec;
		lapsed.tv_sec = second.tv_sec - first.tv_sec;
		printf("\nXronos parallhlou TSP: %d.%06dsec\n\n", (int)lapsed.tv_sec, (int)lapsed.tv_usec);
		
		printf("elaxisto kartas %f\n", find_final_min_and_order(h_min, num_gpus*GRID_SIZE, h_order));
		
		//Apo to index vgazoume to deikth twn pinakwn x kai y gia na mporoume na typwsoume ta shmeia
		int result[maxnpts];
		int stoixeia[maxnpts];
		int b=npts;
		for (i=0; i<npts; i++) stoixeia[i] = i;
		for (i=0; i<npts; i++)	
		{
			result[i]= stoixeia[ h_order[i] ];
			stoixeia[ h_order[i] ] = stoixeia[b-1];
			b--;
		}
		printf("Veltisth diadromh\n");
		for(i=0;i<npts-1;i++){
			printf("(%d,%d)->",x[result[i]],y[result[i]]);
		}
		printf("(%d,%d)\n",x[result[npts-1]],y[result[npts-1]]);
		printf("Pathste ENTER gia synexeia\n");
		getchar();
	}
	//****************************************************************

	float serial_min;

	int *order_x=(int *)malloc( npts*sizeof(int) );
	int *order_y=(int *)malloc( npts*sizeof(int) );
	
	
	gettimeofday(&first, &tzp);
	
	//Ektelesh seiriakou TSP
	serial_min = host_TSP(order_x, order_y);
	
	gettimeofday(&second, &tzp);
	if(first.tv_usec>second.tv_usec){
		second.tv_usec += 1000000;
		second.tv_sec--;
	}
	lapsed.tv_usec = second.tv_usec - first.tv_usec;
	lapsed.tv_sec = second.tv_sec - first.tv_sec;

	printf("\nXronos seiriakou TSP: %d.%06dsec\n\n", (int)lapsed.tv_sec, (int)lapsed.tv_usec);

	printf("permutations:%ld\n\n",perm);
	printf("Veltisth diadromh\n");
	for(i=0;i<npts-1;i++){
		printf("(%d,%d)->",order_x[i],order_y[i]);
	}
	printf("(%d,%d)",order_x[npts-1],order_y[npts-1]);
	printf("\nMinimum distance by CPU:%f\n", serial_min);
	
	//apodesmeysh mnhmhs
	free(h_min);
	free(h_order);
	free(order_x);
	free(order_y);

	cudaThreadExit();	//Exit and clean-up from CUDA launches.
	
	printf("\nEnter gia termatismo");
	getchar();
}


void input(void) 
{
	int i, n;
	printf("Dwse ta shmeia sth swsth seira: x y, me ka8e zeygari se\n");
	printf("ka8e grammh, me keno anamesa\n");
	printf("Ta x y na einai akeraia\n");
	printf("Grapse END gia na stamathsei\n");
	for (n = 0;; n++) {
		gets(answer);
		if (answer[0] == 'E' || answer[0] == 'e') break;
		x[n] = atoi(answer);
		i = 0;
		while (answer[i] != ' ' && answer[i] != '\0')i++;
		y[n] = atoi(answer + i);
	}
	npts = n;
}

float host_TSP(int *order_x, int *order_y){
	int temporderx[maxnpts];
	int tempordery[maxnpts];
	int *tempx=(int *)malloc( npts*sizeof(int) );
	int *tempy=(int *)malloc( npts*sizeof(int) );
	float mindist=0;
	int i;
	unsigned long int a;
	//Epanalhpsh gia ka8e meta8esh
	for (a=0; a<perm; a++)
	{	

		for(i=0; i<npts; i++){
			tempx[i]=x[i];
			tempy[i]=y[i];
		}

		float distance=0;
		int oldx=0;
		int oldy=0;
		int b;
		unsigned long int div=perm;		//edw an htan int, kata thn ektelesh 8a mporouse na petaksei float overflow exception
		for (b=npts; b>0; b--) 
		{	
			div/=b;
			int index = (a/div)%b;
			if(b==npts){
				oldx=tempx[index];
				oldy=tempy[index];
			}
			distance+=sqrt((float)((oldx-tempx[index])*(oldx-tempx[index])+(oldy-tempy[index])*(oldy-tempy[index])));//h apostash apo ena shmeio sto epomeno
			temporderx[npts-b]=tempx[index];
			tempordery[npts-b]=tempy[index];
			oldx=tempx[index];
			oldy=tempy[index];
			tempx[index]=tempx[b-1];
			tempy[index]=tempy[b-1];
		}
		//ka8e fora kratame th diadromh me thn elaxisth olikh apostash
		if(a!=0){
			if(mindist>distance){
				mindist=distance;
				for(i=0;i<npts;i++){
					order_x[i]=temporderx[i];
					order_y[i]=tempordery[i];
				}
			}
		}else{
			mindist=distance;
			for(i=0;i<npts;i++){
				order_x[i]=temporderx[i];
				order_y[i]=tempordery[i];
			}
		}
	}
	free(tempx);
	free(tempy);
	return mindist;
}

//Seiriakh anazhthsh ths mikroterhs timhs se ena pinaka
float find_final_min(float *matrixx, int mege8os){
	float result=matrixx[0];
	int i;
	for(i=0;i<mege8os;i++){
		if(result>matrixx[i]){
			result=matrixx[i];
		}
	}
	return result;
}

//Seiriakh anazhthsh ths mikroterhs timhs se ena pinaka kai diathrhsh twn deiktwn
float find_final_min_and_order(float *matrixx, int mege8os, int *order){
	float result=matrixx[0];
	int i,j;
	for(i=0;i<mege8os;i++){
		if(result>matrixx[i]){
			result=matrixx[i];
			for(j=0;j<npts;j++){
				order[j]=order[i*maxnpts+j];
			}
		}
	}
	return result;
}

//kwdikas CUDA
__global__ void dev_TSP(float *d_min, int *d_order)
{
	
	int tx = threadIdx.x;

	int tid=tx+BLOCK_SIZE*blockIdx.x ;

	int tempx[maxnpts];
	int tempy[maxnpts];

	__shared__ float thread_minimums[BLOCK_SIZE];	//se ka8e 8esh apo8hkeyetai to elaxisto pou briskei ka8e nhma toy BLOCK
	__shared__ int order[maxnpts*BLOCK_SIZE];		//apo8hkeyetai h elaxisth apostash ana nhma
	thread_minimums[tx]=999999999;
	__syncthreads();
	unsigned long int a;
	int i;
	
	for (a=tid*d_num_a + cpu_thread_id*d_perm_per_gpu; a<tid*d_num_a + cpu_thread_id*d_perm_per_gpu +d_num_a; a++)
	{	
		if (a <= d_perm_per_gpu + cpu_thread_id*d_perm_per_gpu) {
			for(i=0; i<d_npts; i++){
				tempx[i]=d_x[i];
				tempy[i]=d_y[i];
			}

			float distance=0;
			int oldx, oldy;
			int b;
			unsigned long int div=d_perm;
			int thread_order[maxnpts];
			for (b=d_npts; b>0; b--)
			{	
				div/=b;
				int index = (a/div)%b;	//an to div htan 0 edw, h 2h karta mporei na epestrefe ton idio pinaka d_min
				if(b==d_npts){
					oldx=tempx[index];
					oldy=tempy[index];
				}
				distance+=sqrt((float)((oldx-tempx[index])*(oldx-tempx[index])+(oldy-tempy[index])*(oldy-tempy[index])));
				thread_order[d_npts-b]=index;

				oldx=tempx[index];
				oldy=tempy[index];
				tempx[index]=tempx[b-1];
				tempy[index]=tempy[b-1];
			}


			if (distance < thread_minimums[tx]){ 
				thread_minimums[tx] = distance;
				for(i=0;i<d_npts;i++){
					order[tx*maxnpts+i]=thread_order[i];
				}
			}
		}
	}
	
	//ypologizei elaxisto gia apo8hkefsh
	__syncthreads();

	//Ypologizei to elaxisto tou BLOCK kai to bazei sthn 8esh pou antistoixei sto block afto sthn global
	//o parakatw algori8mmos doulebei gia dynameis tou 2
	for(unsigned int i=BLOCK_SIZE/2; i>0; i>>=1) 
	{
		if (threadIdx.x < i) 
		{
			if (thread_minimums[threadIdx.x + i] < thread_minimums[tx])
			{
				thread_minimums[threadIdx.x] = thread_minimums[threadIdx.x + i];
				for(unsigned int j=0;j<d_npts;j++){
					order[threadIdx.x*maxnpts+j]=order[(threadIdx.x+i)*maxnpts+j];
				}
			}
		}
		__syncthreads();
	}

	//To result brisketai sthn 8esh 0 tou pinaka
	if (threadIdx.x == 0)
	{
		d_min[blockIdx.x] = thread_minimums[0];
		for(unsigned int j=0;j<d_npts;j++){
			d_order[blockIdx.x*maxnpts+j]=order[j];
		}
	}
}
