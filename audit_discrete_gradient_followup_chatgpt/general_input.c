#include <stdio.h>
#include "snapshot/phylo2vec_fastme.c"
int main(void){int64_t v[3]={0,0,3};double D[16];for(int i=0;i<4;i++)for(int j=0;j<4;j++)D[4*i+j]=i==j?0:1;D[2]=D[8]=10;double out[6];calculate_discrete_gradient_c(v,D,4,out);for(int i=0;i<6;i++)printf("%d %.17g\n",i,out[i]);return 0;}
