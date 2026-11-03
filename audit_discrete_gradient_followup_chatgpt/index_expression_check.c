#include <stdint.h>
#include <stdio.h>
int main(){int32_t n=23172,n_nodes_u=2*n-2,i=n_nodes_u-1;size_t index=(size_t)i*n_nodes_u+i;printf("N=%d diagonal index=%zu\n",n,index);n=32770;int32_t n_pairs=n-1,max_cols=2*n_pairs-1;size_t entries=(size_t)n_pairs*(size_t)max_cols;printf("N=%d gradient entries=%zu\n",n,entries);return index!=2147580963ULL||entries!=2147581953ULL;}
