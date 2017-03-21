#include <stdio.h>
#include <math.h>
#include <cutil_inline.h>
#include <omp.h>
#include <sys/time.h>

#define maxnpts 15	//me 14 exoume to beltisto occupancy logw periorismwn ths shared 

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
//float *d_min;

__constant__ int d_x[maxnpts];
__constant__ int d_y[maxnpts];
__constant__ unsigned long int d_perm;
__constant__ int d_npts;
__constant__ unsigned long int d_ari8mos_a;	//ari8mos epanalhpsewn tou ekswterikou broxou pou 8a keni ka8e nhma
__constant__ int anagnwrist_nhmatos_kme;
__constant__ unsigned long int d_perm_kartas;

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
	input();
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
		if(npts>=8){
			int gpu_id = -1;
			int cpu_thread_id = omp_get_thread_num();	//xarakthristikos ari8mos toy nhmatos tou epeksergasth
			unsigned int num_cpu_threads = omp_get_num_threads();	//sbhsimo;;
			cutilSafeCall(cudaSetDevice(cpu_thread_id));
			cutilSafeCall(cudaGetDevice(&gpu_id));
			printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);
			
			unsigned long int perm_kartas = perm/num_gpus;
			if ((unsigned int)cpu_thread_id < perm % num_gpus) perm_kartas++;		//gia thn periptwsh pou o ari8mos meta8esewn den diaireitai akribws me ton ari8mo twn kartwn
			
			unsigned long int h_ari8mos_a = perm_kartas / (BLOCK_SIZE*GRID_SIZE) +1;	//+1 giati h diaresh den einai teleia synh8ws
			
			// create CUDA event handles
			//cudaEvent_t start_event, stop_event;
			//cutilSafeCall( cudaEventCreate(&start_event) );
			//cutilSafeCall( cudaEventCreate(&stop_event) );
			
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

			cudaMemcpyToSymbol("anagnwrist_nhmatos_kme", &cpu_thread_id, sizeof(int));	//edw an ebaza px unsigned long int 8a epairna kata thn ektelesh: kernel execution failed, invalid argument
			cudaMemcpyToSymbol("d_perm_kartas", &perm_kartas, sizeof(unsigned long int));
			cudaMemcpyToSymbol("d_ari8mos_a", &h_ari8mos_a, sizeof(unsigned long int));
			
			//float xronos_d_tsp;
			//cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed
			// execute the kernel
			
			dev_TSP<<<GRID_SIZE, BLOCK_SIZE>>>(d_min,d_order);
			
			//cudaEventRecord(stop_event, 0);
			//cudaEventSynchronize(stop_event);   // block until the event is actually recorded
			//cutilSafeCall( cudaEventElapsedTime(&xronos_d_tsp, start_event, stop_event) );
			//printf("Xronos ypologismoy pwlhth: %fsec apo karta %d\n", xronos_d_tsp/1000, gpu_id);
			
			// check if kernel execution generated and error
			cutilCheckMsg("Kernel execution failed");
			
			//allocate host memory for the result
			float *ypo_h_min = h_min + cpu_thread_id*GRID_SIZE;
			int *ypo_h_order = h_order + cpu_thread_id*GRID_SIZE*maxnpts;
			cutilSafeCall(cudaMemcpy(ypo_h_min, d_min, GRID_SIZE*sizeof(float), cudaMemcpyDeviceToHost));	//Apo8hkeyw ston h_min to apotelesma ths kartas.
			cutilSafeCall(cudaMemcpy(ypo_h_order, d_order, GRID_SIZE*maxnpts*sizeof(int), cudaMemcpyDeviceToHost));	//Apo8hkeyw ston h_min to apotelesma ths kartas.
			
			//apeleye8erwsh mnhmhs
			cutilSafeCall(cudaFree(d_min));
			cutilSafeCall(cudaFree(d_order));
			
			if(omp_get_thread_num()==0)gettimeofday(&second, &tzp);
		
	//		for (i=0; i<GRID_SIZE; i++)
	//			printf("karta=%d %f\n",gpu_id, ypo_h_min[i]);
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
		//#pragma omp flush (list)  newline
		printf("elaxisto kartas %f\n", find_final_min_and_order(h_min, num_gpus*GRID_SIZE, h_order));
		
		int apotelesma[maxnpts];
		int stoixeia[maxnpts];
		int b=npts;
		for (i=0; i<npts; i++) stoixeia[i] = i;
		for (i=0; i<npts; i++)
		{
			apotelesma[i]= stoixeia[ h_order[i] ];
			stoixeia[ h_order[i] ] = stoixeia[b-1];
			b--;
		}
		printf("Veltisth diadromh\n");
		for(i=0;i<npts-1;i++){
			printf("(%d,%d)->",x[apotelesma[i]],y[apotelesma[i]]);
		}
		printf("(%d,%d)\n",x[apotelesma[npts-1]],y[apotelesma[npts-1]]);
		/*printf("Veltisth diadromh\n");
		for(i=0;i<npts;i++){
			printf(" %d ",apotelesma[i]);
		}*/
		printf("Pathste ENTER gia synexeia\n");
		//fflush(stdout);		//edw gia kapoio logo 8elei fflush alliws 8a prepei na perimenoume na teleiwsei o seirakos algori8mos gia na ektypw8oun ta apotelesmata
		getchar();
	}
	//****************************************************************

//	// create CUDA event handles
//	cudaEvent_t start_event, stop_event;
//	cutilSafeCall( cudaEventCreate(&start_event) );
//	cutilSafeCall( cudaEventCreate(&stop_event) );

//	// allocate device memory
//	float *d_min;
//	cutilSafeCall(cudaMalloc((void**) &d_min, GRID_SIZE*sizeof(float)));

//	//arxikopoihsh gia apostolh sthn constant
//	unsigned long int h_ari8mos_a = perm / (BLOCK_SIZE*GRID_SIZE) +1;

//	float xronos_metaforas_eisodou;
//	cudaEventRecord(start_event, 0);
	// copy host memory to device
	//cutilSafeCall(cudaMemcpy(d_pF, pF, mpF*npF*sizeof(float), cudaMemcpyHostToDevice) );

//	cudaMemcpyToSymbol("d_x", x, npts*sizeof(int));
//	cudaMemcpyToSymbol("d_y", y, npts*sizeof(int));
//	cudaMemcpyToSymbol("d_npts", &npts, sizeof(int));
//	cudaMemcpyToSymbol("d_perm", &perm, sizeof(unsigned long int));
//	cudaMemcpyToSymbol("d_ari8mos_a", &h_ari8mos_a, sizeof(unsigned long int));

//	cudaEventRecord(stop_event, 0);
//	cudaEventSynchronize(stop_event);   // block until the event is actually recorded
//	cutilSafeCall( cudaEventElapsedTime(&xronos_metaforas_eisodou, start_event, stop_event) );
//	printf("\n\nXronos metaforas twn dedomenwn eisodoy sthn karta: %fsec\n", xronos_metaforas_eisodou/1000);

//	printf("\nRun Kernel...\n\n");

//	float xronos_d_tsp;
//	cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed

//	// execute the kernel
//	dev_TSP<<< GRID_SIZE, BLOCK_SIZE >>>(d_min);

//	cudaEventRecord(stop_event, 0);
//	cudaEventSynchronize(stop_event);   // block until the event is actually recorded
//	cutilSafeCall( cudaEventElapsedTime(&xronos_d_tsp, start_event, stop_event) );
//	printf("Xronos ypologismoy pwlhth: %fsec\n", xronos_d_tsp/1000);

//	// check if kernel execution generated and error
//	cutilCheckMsg("Kernel execution failed");

//	//cudaThreadSynchronize();	//Wait for compute-device to finish.

//	// allocate host memory for the result
//	float *h_min = (float*) malloc(GRID_SIZE*sizeof(float));
//	float xronos_lhpshs_eksodou;
//	cudaEventRecord(start_event, 0);

//	cutilSafeCall(cudaMemcpy(h_min, d_min, GRID_SIZE*sizeof(float), cudaMemcpyDeviceToHost));	//Apo8hkeyw ston h_min to apotelesma ths kartas.

//	cudaEventRecord(stop_event, 0);
//	cudaEventSynchronize(stop_event);   // block until the event is actually recorded
//	cutilSafeCall( cudaEventElapsedTime(&xronos_lhpshs_eksodou, start_event, stop_event) );
//	printf("\nXronos poy apaiteitai gia thn lhpsh twn apotelesmatwn apo thn karta: %fsec\n", xronos_lhpshs_eksodou/1000);

	float elax_seiriakou;

	int *order_x=(int *)malloc( npts*sizeof(int) );
	int *order_y=(int *)malloc( npts*sizeof(int) );
	
	
	gettimeofday(&first, &tzp);
	
	elax_seiriakou = host_TSP(order_x, order_y);
	
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
	printf("\nMinimum distance by CPU:%f\n", elax_seiriakou);
	//printf("Minimum distance by GPU:%f\n", find_final_min(h_min, num_gpus*GRID_SIZE));
	
	free(h_min);
	free(h_order);
	free(order_x);
	free(order_y);
	//cudaEventDestroy(start_event);
	//cudaEventDestroy(stop_event);
	//cutilSafeCall(cudaFree(d_min));

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
	for (a=0; a<perm; a++)
	{	
		//printf("Permutation number:%d ", a);
		for(i=0; i<npts; i++){
			tempx[i]=x[i];
			tempy[i]=y[i];
		}

		float distance=0;
		int oldx=0;
		int oldy=0;
		int b, div=perm;
		for (b=npts; b>0; b--) 
		{	
			div/=b;
			int index = (a/div)%b;
			if(b==npts){
				oldx=tempx[index];
				oldy=tempy[index];
			}
			distance+=sqrt((float)((oldx-tempx[index])*(oldx-tempx[index])+(oldy-tempy[index])*(oldy-tempy[index])));
			//printf("(%d,%d)",tempx[index],tempy[index] );
			//printf("index=%d (%d,%d)",index,tempx[index],tempy[index] );
			temporderx[npts-b]=tempx[index];
			tempordery[npts-b]=tempy[index];
			oldx=tempx[index];
			oldy=tempy[index];
			tempx[index]=tempx[b-1];
			tempy[index]=tempy[b-1];
		}
		//printf("\n");
		//printf("Total distance %f \n",distance);
		//printf("%d %d %d",order[0],order[1],order[2]);
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

float find_final_min(float *pinakas, int mege8os){
	float result=pinakas[0];
	int i;
	for(i=0;i<mege8os;i++){
		if(result>pinakas[i]){
			result=pinakas[i];
		}
	}
	return result;
}

float find_final_min_and_order(float *pinakas, int mege8os, int *order){
	float result=pinakas[0];
	int i,j;
	for(i=0;i<mege8os;i++){
		if(result>pinakas[i]){
			result=pinakas[i];
			for(j=0;j<npts;j++){
				order[j]=order[i*maxnpts+j];
			}
		}
	}
	return result;
}

__global__ void dev_TSP(float *d_min, int *d_order)
{
	// Thread index -                       thread     block
	int tx = threadIdx.x;

	int tid=tx+BLOCK_SIZE*blockIdx.x ;

	int tempx[maxnpts];
	int tempy[maxnpts];

	__shared__ float elaxista_nhmatwn[BLOCK_SIZE];	//se ka8e 8esh apo8hkeyetai to elaxisto pou briskei ka8e nhma toy BLOCK
	__shared__ int order[maxnpts*BLOCK_SIZE];
	elaxista_nhmatwn[tx]=999999999;
	__syncthreads();
	unsigned long int a;
	int i;
	//for (a=tid*d_ari8mos_a + anagnwrist_nhmatos_kme*GRID_SIZE*d_ari8mos_a; a<tid*d_ari8mos_a + anagnwrist_nhmatos_kme*GRID_SIZE*d_ari8mos_a +d_ari8mos_a; a++)
	for (a=tid*d_ari8mos_a + anagnwrist_nhmatos_kme*d_perm_kartas; a<tid*d_ari8mos_a + anagnwrist_nhmatos_kme*d_perm_kartas +d_ari8mos_a; a++)
	{	
		if (a <= d_perm_kartas + anagnwrist_nhmatos_kme*d_perm_kartas) {
			//printf("Permutation number:%d ", a);
			for(i=0; i<d_npts; i++){
				tempx[i]=d_x[i];
				tempy[i]=d_y[i];
			}

			float distance=0;
			int oldx, oldy;
			int b, div=d_perm;
			int thread_order[maxnpts];
			for (b=d_npts; b>0; b--)
			{	
				div/=b;
				int index = (a/div)%b;
				if(b==d_npts){
					oldx=tempx[index];
					oldy=tempy[index];
				}
				distance+=sqrt((float)((oldx-tempx[index])*(oldx-tempx[index])+(oldy-tempy[index])*(oldy-tempy[index])));
				thread_order[d_npts-b]=index;
				//printf("index=%d (%d,%d)",index,tempx[index],tempy[index] );
				//order[npts-b]=b;

				oldx=tempx[index];
				oldy=tempy[index];
				tempx[index]=tempx[b-1];
				tempy[index]=tempy[b-1];
			}
			//printf("Total distance %f \n",distance);
			//printf("%d %d %d",order[0],order[1],order[2]);

			if (distance < elaxista_nhmatwn[tx]){ 
				elaxista_nhmatwn[tx] = distance;
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
			if (elaxista_nhmatwn[threadIdx.x + i] < elaxista_nhmatwn[tx])
			{
				//elaxista_nhmatwn[tx] = elaxista_nhmatwn[threadIdx.x + i];
				elaxista_nhmatwn[threadIdx.x] = elaxista_nhmatwn[threadIdx.x + i];
				for(unsigned int j=0;j<d_npts;j++){
					order[threadIdx.x*maxnpts+j]=order[(threadIdx.x+i)*maxnpts+j];
				}
			}
		}
		__syncthreads();
	}

	//To apotelesma brisketai sthn 8esh 0 tou pinaka
	if (threadIdx.x == 0)
	{
		//atomicMin(&d_min[blockIdx.x], elaxista_nhmatwn[0]);	//doulebei mono gia akeraious
		//d_min[anagnwrist_nhmatos_kme*GRID_SIZE + blockIdx.x] = elaxista_nhmatwn[0];
		d_min[blockIdx.x] = elaxista_nhmatwn[0];
		for(unsigned int j=0;j<d_npts;j++){
			d_order[blockIdx.x*maxnpts+j]=order[j];
		}
	}
}